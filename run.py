#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime

import uuid
import glob
from pathlib import Path
import fire

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from ignite.contrib.handlers import ProgressBar, CustomPeriodicEvent, param_scheduler
from ignite.engine import (Engine, Events)
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, RunningAverage, Precision, Recall
from ignite.utils import convert_tensor
from tabulate import tabulate

import dataset
import models
import utils
import metrics
import losses

DEVICE = 'cpu'
if torch.cuda.is_available(
        ):
    DEVICE = 'cuda'
    # Without results are slightly inconsistent
    torch.backends.cudnn.deterministic = True
DEVICE = torch.device(DEVICE)


class Runner(object):
    """Main class to run experiments with e.g., train and evaluate"""
    def __init__(self, seed=42):
        """__init__

        :param config: YAML config file
        :param **kwargs: Overwrite of yaml config
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _forward(model, batch):
        inputs, targets, filenames = batch
        inputs, targets = convert_tensor(inputs,
                                         device=DEVICE,
                                         non_blocking=True), convert_tensor(
                                             targets.float(),
                                             device=DEVICE,
                                             non_blocking=True)
        clip_level_output, frame_level_output = model(inputs)
        return clip_level_output, frame_level_output, targets

    @staticmethod
    def _negative_loss(engine):
        return -engine.state.metrics['Loss']

    def train(self, config, **kwargs):
        """Trains a given model specified in the config file or passed as the --model parameter.
        All options in the config file can be overwritten as needed by passing --PARAM
        Options with variable lengths ( e.g., kwargs can be passed by --PARAM '{"PARAM1":VAR1, "PARAM2":VAR2}'

        :param config: yaml config file
        :param **kwargs: parameters to overwrite yaml config
        """

        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        outputdir = os.path.join(
            config_parameters['outputpath'], config_parameters['model'],
            "{}_{}".format(
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%m'),
                uuid.uuid1().hex))
        # Create base dir
        Path(outputdir).mkdir(exist_ok=True, parents=True)

        logger = utils.getfile_outlogger(os.path.join(outputdir, 'train.log'))
        logger.info("Storing files in {}".format(outputdir))
        # utils.pprint_dict
        utils.pprint_dict(config_parameters, logger.info)
        logger.info("Running on device {}".format(DEVICE))
        labels_df = pd.read_csv(config_parameters['label'],
                                sep='\s+').convert_dtypes()
        # In case of ave dataset where index is int, we change the
        # absolute name to relname
        if not np.all(labels_df['filename'].str.isnumeric()):
            labels_df.loc[:, 'filename'] = labels_df['filename'].apply(
                os.path.basename)
        encoder = utils.train_labelencoder(labels=labels_df['event_labels'])
        # These labels are useless, only for mode == stratified
        label_array, _ = utils.encode_labels(labels_df['event_labels'],
                                             encoder)
        if 'cv_label' in config_parameters:
            cv_df = pd.read_csv(config_parameters['cv_label'],
                                sep='\s+').convert_dtypes()
            if not np.all(cv_df['filename'].str.isnumeric()):
                cv_df.loc[:, 'filename'] = cv_df['filename'].apply(
                    os.path.basename)
            train_df = labels_df
            logger.info(
                f"Using CV labels from {config_parameters['cv_label']}")
        else:
            train_df, cv_df = utils.split_train_cv(
                labels_df, y=label_array, **config_parameters['data_args'])

        if 'cv_data' in config_parameters:
            cv_data = config_parameters['cv_data']
            logger.info(f"Using CV data {config_parameters['cv_data']}")
        else:
            cv_data = config_parameters['data']

        train_label_array, _ = utils.encode_labels(train_df['event_labels'],
                                                   encoder)
        cv_label_array, _ = utils.encode_labels(cv_df['event_labels'], encoder)

        transform = utils.parse_transforms(config_parameters['transforms'])
        utils.pprint_dict({'Classes': encoder.classes_},
                          logger.info,
                          formatter='pretty')
        torch.save(encoder, os.path.join(outputdir, 'run_encoder.pth'))
        torch.save(config_parameters, os.path.join(outputdir,
                                                   'run_config.pth'))
        logger.info("Transforms:")
        utils.pprint_dict(transform, logger.info, formatter='pretty')
        # For Unbalanced Audioset, this is true
        if 'sampler' in config_parameters and config_parameters[
                'sampler'] == 'MultiBalancedSampler':
            # Training sampler that oversamples the dataset to be roughly equally sized
            # Calcualtes mean over multiple instances, rather useful when number of classes
            # is large
            train_sampler = dataset.MultiBalancedSampler(
                train_label_array,
                num_samples=1 * train_label_array.shape[0],
                replacement=True)
            sampling_kwargs = {"shuffle": False, "sampler": train_sampler}
        elif 'sampler' in config_parameters and config_parameters[
                'sampler'] == 'MinimumOccupancySampler':
            # Asserts that each "batch" contains at least one instance
            train_sampler = dataset.MinimumOccupancySampler(
                train_label_array, sampling_mode='same')
            sampling_kwargs = {"shuffle": False, "sampler": train_sampler}
        else:
            sampling_kwargs = {"shuffle": True}

        logger.info("Using Sampler {}".format(sampling_kwargs))

        trainloader = dataset.getdataloader(
            {
                'filename': train_df['filename'].values,
                'encoded': train_label_array
            },
            config_parameters['data'],
            transform=transform,
            batch_size=config_parameters['batch_size'],
            colname=config_parameters['colname'],
            num_workers=config_parameters['num_workers'],
            **sampling_kwargs)

        cvdataloader = dataset.getdataloader(
            {
                'filename': cv_df['filename'].values,
                'encoded': cv_label_array
            },
            cv_data,
            transform=None,
            shuffle=False,
            colname=config_parameters['colname'],
            batch_size=config_parameters['batch_size'],
            num_workers=config_parameters['num_workers'])
        model = getattr(models, config_parameters['model'],
                        'CRNN')(inputdim=trainloader.dataset.datadim,
                                outputdim=len(encoder.classes_),
                                **config_parameters['model_args'])
        if 'pretrained' in config_parameters and config_parameters[
                'pretrained'] is not None:
            models.load_pretrained(model,
                                   config_parameters['pretrained'],
                                   outputdim=len(encoder.classes_))
            logger.info("Loading pretrained model {}".format(
                config_parameters['pretrained']))

        model = model.to(DEVICE)
        if config_parameters['optimizer'] == 'AdaBound':
            try:
                import adabound
                optimizer = adabound.AdaBound(
                    model.parameters(), **config_parameters['optimizer_args'])
            except ImportError:
                config_parameters['optimizer'] = 'Adam'
                config_parameters['optimizer_args'] = {}
        else:
            optimizer = getattr(
                torch.optim,
                config_parameters['optimizer'],
            )(model.parameters(), **config_parameters['optimizer_args'])

        utils.pprint_dict(optimizer, logger.info, formatter='pretty')
        utils.pprint_dict(model, logger.info, formatter='pretty')
        if DEVICE.type != 'cpu' and torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs!".format(torch.cuda.device_count()))
            model = torch.nn.DataParallel(model)
        criterion = getattr(losses, config_parameters['loss'])().to(DEVICE)

        def _train_batch(_, batch):
            model.train()
            with torch.enable_grad():
                optimizer.zero_grad()
                output = self._forward(
                    model, batch)  # output is tuple (clip, frame, target)
                loss = criterion(*output)
                loss.backward()
                # Single loss
                optimizer.step()
                return loss.item()

        def _inference(_, batch):
            model.eval()
            with torch.no_grad():
                return self._forward(model, batch)

        def thresholded_output_transform(output):
            # Output is (clip, frame, target)
            y_pred, _, y = output
            y_pred = torch.round(y_pred)
            return y_pred, y

        precision = Precision(thresholded_output_transform, average=False)
        recall = Recall(thresholded_output_transform, average=False)
        f1_score = (precision * recall * 2 / (precision + recall)).mean()
        metrics = {
            'Loss': losses.Loss(
                criterion),  #reimplementation of Loss, supports 3 way loss 
            'Precision': Precision(thresholded_output_transform),
            'Recall': Recall(thresholded_output_transform),
            'Accuracy': Accuracy(thresholded_output_transform),
            'F1': f1_score,
        }
        train_engine = Engine(_train_batch)
        inference_engine = Engine(_inference)
        for name, metric in metrics.items():
            metric.attach(inference_engine, name)

        def compute_metrics(engine):
            inference_engine.run(cvdataloader)
            results = inference_engine.state.metrics
            output_str_list = [
                "Validation Results - Epoch : {:<5}".format(engine.state.epoch)
            ]
            for metric in metrics:
                output_str_list.append("{} {:<5.2f}".format(
                    metric, results[metric]))
            logger.info(" ".join(output_str_list))

        pbar = ProgressBar(persist=False)
        pbar.attach(train_engine)

        if 'itercv' in config_parameters and config_parameters[
                'itercv'] is not None:
            train_engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=config_parameters['itercv']),
                compute_metrics)
        train_engine.add_event_handler(Events.EPOCH_COMPLETED, compute_metrics)

        # Default scheduler is using patience=3, factor=0.1
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **config_parameters['scheduler_args'])

        @inference_engine.on(Events.EPOCH_COMPLETED)
        def update_reduce_on_plateau(engine):
            logger.info(f"Scheduling epoch {engine.state.epoch}")
            val_loss = engine.state.metrics['Loss']
            if 'ReduceLROnPlateau' == scheduler.__class__.__name__:
                scheduler.step(val_loss)
            else:
                scheduler.step()

        early_stop_handler = EarlyStopping(
            patience=config_parameters['early_stop'],
            score_function=self._negative_loss,
            trainer=train_engine)
        inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           early_stop_handler)
        if config_parameters['save'] == 'everyepoch':
            checkpoint_handler = ModelCheckpoint(outputdir,
                                                 'run',
                                                 n_saved=5,
                                                 require_empty=False)
            train_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                           checkpoint_handler, {
                                               'model': model,
                                           })
            train_engine.add_event_handler(
                Events.ITERATION_COMPLETED(every=config_parameters['itercv']),
                checkpoint_handler, {
                    'model': model,
                })
        else:
            checkpoint_handler = ModelCheckpoint(
                outputdir,
                'run',
                n_saved=1,
                require_empty=False,
                score_function=self._negative_loss,
                global_step_transform=global_step_from_engine(
                    train_engine),  # Just so that model is saved with epoch...
                score_name='loss')
            inference_engine.add_event_handler(Events.EPOCH_COMPLETED,
                                               checkpoint_handler, {
                                                   'model': model,
                                               })

        train_engine.run(trainloader, max_epochs=config_parameters['epochs'])
        return outputdir

    def evaluate(
            self,
            experiment_path: str,
            pred_file='hard_predictions_{}.txt',
            tag_file='tagging_predictions_{}.txt',
            event_file='event_{}.txt',
            segment_file='segment_{}.txt',
            class_result_file='class_result_{}.txt',
            time_ratio=10. / 500,
            postprocessing='double',
            threshold=None,
            window_size=None,
            save_seq=False,
            sed_eval=True,  # Do evaluation on sound event detection ( time stamps, segemtn/evaluation based)
            **kwargs):
        """evaluate

        :param experiment_path: Path to already trained model using train
        :type experiment_path: str
        :param pred_file: Prediction output file, put into experiment dir
        :param time_resolution: Resolution in time (1. represents the model resolution)
        :param **kwargs: Overwrite standard args, please pass `data` and `label`
        """
        # Update config parameters with new kwargs

        config = torch.load(list(Path(f'{experiment_path}').glob("run_config*"))[0], map_location='cpu')
        # Use previous config, but update data such as kwargs
        config_parameters = dict(config, **kwargs)
        # Default columns to search for in data
        config_parameters.setdefault('colname', ('filename', 'encoded'))
        model_parameters = torch.load(
            glob.glob("{}/run_model*".format(experiment_path))[0],
            map_location=lambda storage, loc: storage)
        encoder = torch.load(glob.glob(
            '{}/run_encoder*'.format(experiment_path))[0],
                             map_location=lambda storage, loc: storage)
        strong_labels_df = pd.read_csv(config_parameters['label'], sep='\t')

        # Evaluation is done via the filenames, not full paths
        if not np.issubdtype(strong_labels_df['filename'].dtype, np.number):
            strong_labels_df['filename'] = strong_labels_df['filename'].apply(
                os.path.basename)
        if 'audiofilepath' in strong_labels_df.columns:  # In case of ave dataset, the audiofilepath column is the main column
            strong_labels_df['audiofilepath'] = strong_labels_df[
                'audiofilepath'].apply(os.path.basename)
            colname = 'audiofilepath'  # AVE
        else:
            colname = 'filename'  # Dcase etc.
        # Problem is that we iterate over the strong_labels_df, which is ambigious
        # In order to conserve some time and resources just reduce strong_label to weak_label format
        weak_labels_df = strong_labels_df.groupby(
            colname)['event_label'].unique().apply(
                tuple).to_frame().reset_index()
        if "event_labels" in strong_labels_df.columns:
            assert False, "Data with the column event_labels are used to train not to evaluate"
        weak_labels_array, encoder = utils.encode_labels(
            labels=weak_labels_df['event_label'], encoder=encoder)
        dataloader = dataset.getdataloader(
            {
                'filename': weak_labels_df['filename'].values,
                'encoded': weak_labels_array,
            },
            config_parameters['data'],
            batch_size=1,
            shuffle=False,
            colname=config_parameters[
                'colname']  # For other datasets with different key names
        )
        model = getattr(models, config_parameters['model'])(
            inputdim=dataloader.dataset.datadim,
            outputdim=len(encoder.classes_),
            **config_parameters['model_args'])
        model.load_state_dict(model_parameters)
        model = model.to(DEVICE).eval()
        time_predictions, clip_predictions = [], []
        sequences_to_save = []
        mAP_pred, mAP_tar = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, unit='file', leave=False):
                _, target, filenames = batch
                clip_pred, pred, _ = self._forward(model, batch)
                clip_pred = clip_pred.cpu().detach().numpy()
                mAP_tar.append(target.numpy().squeeze(0))
                mAP_pred.append(clip_pred.squeeze(0))
                pred = pred.cpu().detach().numpy()
                if postprocessing == 'median':
                    if threshold is None:
                        thres = 0.5
                    else:
                        thres = threshold
                    if window_size is None:
                        window_size = 1
                    filtered_pred = utils.median_filter(
                        pred, window_size=window_size, threshold=thres)
                    decoded_pred = utils.decode_with_timestamps(
                        encoder, filtered_pred)


                elif postprocessing == 'cATP-SDS':
                    # cATP-SDS postprocessing uses an "Optimal" configurations, assumes we have a prior
                    # Values are taken from the Surface Disentange paper
                    # Classes are (DCASE2018 only)
                    # ['Alarm_bell_ringing' 'Blender' 'Cat' 'Dishes' 'Dog'
                    # 'Electric_shaver_toothbrush' 'Frying' 'Running_water' 'Speech'
                    # 'Vacuum_cleaner']
                    assert pred.shape[
                        -1] == 10, "Only supporting DCASE2018 for now"
                    if threshold is None:
                        thres = 0.5
                    else:
                        thres = threshold
                    if window_size is None:
                        window_size = [17, 42, 17, 9, 16, 74, 85, 64, 18, 87]
                    # P(y|x) > alpha
                    clip_pred = utils.binarize(clip_pred, threshold=thres)
                    pred = pred * clip_pred
                    filtered_pred = np.zeros_like(pred)

                    # class specific filtering via median filter
                    for cl in range(pred.shape[-1]):
                        # Median filtering also applies thresholding
                        filtered_pred[..., cl] = utils.median_filter(
                            pred[..., cl],
                            window_size=window_size[cl],
                            threshold=thres)
                    decoded_pred = utils.decode_with_timestamps(
                        encoder, filtered_pred)

                elif postprocessing == 'double':
                    # Double thresholding as described in
                    # https://arxiv.org/abs/1904.03841
                    if threshold is None:
                        hi_thres, low_thres = (0.75, 0.2)
                    else:
                        hi_thres, low_thres = threshold
                    filtered_pred = utils.double_threshold(pred,
                                                           high_thres=hi_thres,
                                                           low_thres=low_thres)
                    decoded_pred = utils.decode_with_timestamps(
                        encoder, filtered_pred)

                elif postprocessing == 'triple':
                    # Triple thresholding as described in
                    # Using frame level + clip level predictions
                    if threshold is None:
                        clip_thres, hi_thres, low_thres = (0.5, 0.75, 0.2)
                    else:
                        clip_thres, hi_thres, low_thres = threshold

                    clip_pred = utils.binarize(clip_pred, threshold=clip_thres)
                    # Apply threshold to
                    pred = clip_pred * pred
                    filtered_pred = utils.double_threshold(pred,
                                                           high_thres=hi_thres,
                                                           low_thres=low_thres)
                    decoded_pred = utils.decode_with_timestamps(
                        encoder, filtered_pred)

                for num_batch in range(len(decoded_pred)):
                    filename = filenames[num_batch]
                    cur_pred = pred[num_batch]
                    cur_clip = clip_pred[num_batch].reshape(1, -1)
                    # Clip predictions, independent of per-frame predictions
                    bin_clips = utils.binarize(cur_clip)
                    # Binarize with default threshold 0.5 For clips
                    bin_clips = encoder.inverse_transform(
                        bin_clips.reshape(1,
                                          -1))[0]  # 0 since only single sample
                    # Add each label individually into list
                    for clip_label in bin_clips:
                        clip_predictions.append({
                            'filename': filename,
                            'event_label': clip_label,
                        })
                    # Save each frame output, for later visualization
                    if save_seq:
                        labels = weak_labels_df.loc[weak_labels_df['filename']
                                                    == filename]['event_label']
                        to_save_df = pd.DataFrame(pred[num_batch],
                                                  columns=encoder.classes_)

                        # True labels
                        to_save_df.rename({'variable': 'event'},
                                          axis='columns',
                                          inplace=True)
                        to_save_df['filename'] = filename
                        to_save_df['pred_labels'] = np.array(labels).repeat(
                            len(to_save_df))
                        sequences_to_save.append(to_save_df)
                    label_prediction = decoded_pred[num_batch]
                    for event_label, onset, offset in label_prediction:
                        time_predictions.append({
                            'filename': filename,
                            'event_label': event_label,
                            'onset': onset,
                            'offset': offset
                        })

        assert len(time_predictions) > 0, "No outputs, lower threshold?"
        pred_df = pd.DataFrame(
            time_predictions,
            columns=['filename', 'event_label', 'onset', 'offset'])
        clip_pred_df = pd.DataFrame(
            clip_predictions,
            columns=['filename', 'event_label', 'probability'])
        test_data_filename = os.path.splitext(
            os.path.basename(config_parameters['label']))[0]

        if save_seq:
            pd.concat(sequences_to_save).to_csv(os.path.join(
                experiment_path, 'probabilities.csv'),
                                                index=False,
                                                sep='\t',
                                                float_format="%.4f")

        pred_df = utils.predictions_to_time(pred_df, ratio=time_ratio)
        if pred_file:
            pred_df.to_csv(os.path.join(experiment_path,
                                        pred_file.format(test_data_filename)),
                           index=False,
                           sep="\t")
        tagging_df = metrics.audio_tagging_results(strong_labels_df, pred_df)
        clip_tagging_df = metrics.audio_tagging_results(
            strong_labels_df, clip_pred_df)
        print("Tagging Classwise Result: \n{}".format(
            tabulate(clip_tagging_df,
                     headers='keys',
                     showindex=False,
                     tablefmt='github')))
        print("mAP: {}".format(
            metrics.mAP(np.array(mAP_tar), np.array(mAP_pred))))
        if tag_file:
            clip_tagging_df.to_csv(os.path.join(
                experiment_path, tag_file.format(test_data_filename)),
                                   index=False,
                                   sep='\t')

        if sed_eval:
            event_result, segment_result = metrics.compute_metrics(
                strong_labels_df, pred_df, time_resolution=1.0)
            print("Event Based Results:\n{}".format(event_result))
            event_results_dict = event_result.results_class_wise_metrics()
            class_wise_results_df = pd.DataFrame().from_dict({
                f: event_results_dict[f]['f_measure']
                for f in event_results_dict.keys()
            }).T
            class_wise_results_df.to_csv(os.path.join(
                experiment_path, class_result_file.format(test_data_filename)),
                                         sep='\t')
            print("Class wise F1-Macro:\n{}".format(
                tabulate(class_wise_results_df,
                         headers='keys',
                         tablefmt='github')))
            if event_file:
                with open(
                        os.path.join(experiment_path,
                                     event_file.format(test_data_filename)),
                        'w') as wp:
                    wp.write(event_result.__str__())
            print("=" * 100)
            print(segment_result)
            if segment_file:
                with open(
                        os.path.join(experiment_path,
                                     segment_file.format(test_data_filename)),
                        'w') as wp:
                    wp.write(segment_result.__str__())
            event_based_results = pd.DataFrame(
                event_result.results_class_wise_average_metrics()['f_measure'],
                index=['event_based'])
            segment_based_results = pd.DataFrame(
                segment_result.results_class_wise_average_metrics()
                ['f_measure'],
                index=['segment_based'])
            result_quick_report = pd.concat((
                event_based_results,
                segment_based_results,
            ))
            # Add two columns

            tagging_macro_f1, tagging_macro_pre, tagging_macro_rec = tagging_df.loc[
                tagging_df['label'] == 'macro'].values[0][1:]
            static_tagging_macro_f1, static_tagging_macro_pre, static_tagging_macro_rec = clip_tagging_df.loc[
                clip_tagging_df['label'] == 'macro'].values[0][1:]
            result_quick_report.loc['Time Tagging'] = [
                tagging_macro_f1, tagging_macro_pre, tagging_macro_rec
            ]
            result_quick_report.loc['Clip Tagging'] = [
                static_tagging_macro_f1, static_tagging_macro_pre,
                static_tagging_macro_rec
            ]
            with open(
                    os.path.join(
                        experiment_path,
                        'quick_report_{}.md'.format(test_data_filename)),
                    'w') as wp:
                print(tabulate(result_quick_report,
                               headers='keys',
                               tablefmt='github'),
                      file=wp)

            print("Quick Report: \n{}".format(
                tabulate(result_quick_report,
                         headers='keys',
                         tablefmt='github')))

    def train_evaluate(self, config, test_data, test_label, **kwargs):
        experiment_path = self.train(config, **kwargs)
        from h5py import File
        # Get the output time-ratio factor from the model
        model_parameters = torch.load(
            glob.glob("{}/run_model*".format(experiment_path))[0],
            map_location=lambda storage, loc: storage)
        config_param = torch.load(glob.glob(
            "{}/run_config*".format(experiment_path))[0],
                                  map_location=lambda storage, loc: storage)
        encoder = torch.load(glob.glob(
            '{}/run_encoder*'.format(experiment_path))[0],
                             map_location=lambda storage, loc: storage)
        # Dummy to calculate the pooling factor a bit dynamic
        with File(test_data, 'r') as store:
            timedim, datadim = next(iter(store.values())).shape
        model = getattr(models,
                        config_param['model'])(inputdim=datadim,
                                               outputdim=len(encoder.classes_),
                                               **config_param['model_args'])
        model.load_state_dict(model_parameters)
        dummy = torch.randn(1, timedim, datadim)
        _, time_out = model(dummy)
        time_ratio = max(0.02, 0.02 * np.round(timedim / time_out.shape[1]))
        # Parse for evaluation and update original values such as
        # --data
        # --label
        config_parameters = utils.parse_config_or_kwargs(config, **kwargs)
        threshold = config_parameters.get('threshold', None)
        postprocessing = config_parameters.get('postprocessing', 'double')
        window_size = config_parameters.get('window_size', None)
        self.evaluate(experiment_path,
                      label=test_label,
                      data=test_data,
                      time_ratio=time_ratio,
                      postprocessing=postprocessing,
                      threshold=threshold,
                      window_size=window_size)


    def predict(self, experiment_path, output_csv, thres=0.75,
                **kwargs):  # overwrite --data

        from h5py import File
        from sklearn.preprocessing import binarize
        from tqdm import tqdm
        config = torch.load(glob.glob(
            "{}/run_config*".format(experiment_path))[0],
                            map_location=lambda storage, loc: storage)
        config_parameters = dict(config, **kwargs)
        model_parameters = torch.load(
            glob.glob("{}/run_model*".format(experiment_path))[0],
            map_location=lambda storage, loc: storage)
        encoder = torch.load(glob.glob(
            '{}/run_encoder*'.format(experiment_path))[0],
                             map_location=lambda storage, loc: storage)

        predictions = []
        with File(config_parameters['data'],
                       'r') as input_store, torch.no_grad(), tqdm(
                           total=len(input_store)) as pbar:
            inputdim = next(iter(input_store.values())).shape[-1]
            model = getattr(models, config_parameters['model'])(
                inputdim=inputdim,
                outputdim=len(encoder.classes_),
                **config_parameters['model_args'])
            model.load_state_dict(model_parameters)
            model = model.to(DEVICE).eval()
            for fname, sample in input_store.items():
                if sample.ndim > 1:  # Global mean and Global_var might also be there
                    sample = torch.as_tensor(sample[()]).unsqueeze(0).to(
                        DEVICE)  # batch + channel
                    decision, _ = model(sample)
                    decision = binarize(decision.to('cpu'), threshold=thres)
                    pred_labels = encoder.inverse_transform(decision)[0]
                    pbar.set_postfix(labels=pred_labels, file=fname)
                    if len(pred_labels) > 0:
                        predictions.append({
                            'filename':
                            fname,
                            'event_labels':
                            ",".join(pred_labels)
                        })
                pbar.update()

        df = pd.DataFrame(predictions)
        df.to_csv(output_csv, sep='\t', index=False)

    def evaluate_tagging(self,
                         experiment_path: str,
                         tag_file='tagging_predictions_{}.txt',
                         **kwargs):
        exppath = Path(experiment_path)
        if exppath.is_file(): # Best model passed!
            model_parameters = torch.load(
                str(exppath),
                map_location=lambda storage, loc: storage)
            experiment_path = exppath.parent # Just set upper path as default
        else:
            model_parameters = torch.load(
                glob.glob("{}/run_model*".format(experiment_path))[0],
                map_location=lambda storage, loc: storage)
        config = torch.load(glob.glob(
            "{}/run_config*".format(experiment_path))[0],
                            map_location=lambda storage, loc: storage)
        logger = utils.getfile_outlogger(None)
        # Use previous config, but update data such as kwargs
        config_parameters = dict(config, **kwargs)
        # Default columns to search for in data
        config_parameters.setdefault('colname', ('filename', 'encoded'))
        encoder = torch.load(glob.glob(
            '{}/run_encoder*'.format(experiment_path))[0],
                             map_location=lambda storage, loc: storage)

        test_data_filename = os.path.splitext(
            os.path.basename(config_parameters['label']))[0]
        strong_labels_df = pd.read_csv(config_parameters['label'], sep='\s+')
        # Evaluation is done via the filenames, not full paths
        if not np.issubdtype(strong_labels_df['filename'].dtype, np.number):
            strong_labels_df['filename'] = strong_labels_df['filename'].apply(
                os.path.basename)
        if 'audiofilepath' in strong_labels_df.columns:  # In case of ave dataset, the audiofilepath column is the main column
            strong_labels_df['audiofilepath'] = strong_labels_df[
                'audiofilepath'].apply(os.path.basename)
            colname = 'audiofilepath'  # AVE
        else:
            colname = 'filename'  # Dcase etc.
        weak_labels_df = strong_labels_df.groupby(
            colname)['event_label'].unique().apply(
                tuple).to_frame().reset_index()
        if "event_labels" in strong_labels_df.columns:
            assert False, "Data with the column event_labels are used to train not to evaluate"
        weak_labels_array, encoder = utils.encode_labels(
            labels=weak_labels_df['event_label'], encoder=encoder)
        # assert (weak_labels_df['encoded'].apply(lambda x: sum(x)) >
        # 0).all(), "No targets found, is the encoder maybe not right?"
        for k, v in config_parameters.items():
            logger.info(f"{k}:{v}")
        dataloader = dataset.getdataloader(
            {
                'filename': weak_labels_df['filename'].values,
                'encoded': weak_labels_array
            },
            config_parameters['data'],
            batch_size=1,
            shuffle=False,
            colname=config_parameters[
                'colname'],  # For other datasets with different key names
            num_workers=3,
        )
        model = getattr(models, config_parameters['model'])(
            inputdim=dataloader.dataset.datadim,
            outputdim=len(encoder.classes_),
            **config_parameters['model_args'])
        model.load_state_dict(model_parameters)
        model = model.to(DEVICE).eval()
        y_pred, y_true = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, unit='file', leave=False):
                _, target, filenames = batch
                clip_pred, _, _ = self._forward(model, batch)
                clip_pred = clip_pred.cpu().detach().numpy()
                y_pred.append(clip_pred)
                y_true.append(target.numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        mAP = np.nan_to_num(metrics.mAP(y_true, y_pred))
        auc = np.nan_to_num(metrics.roc(y_true, y_pred))
        with open(
                os.path.join(experiment_path,
                             tag_file.format(test_data_filename)), 'w') as wp:
            print(f"mAP:{mAP.mean():.3f}", file=wp)
            print(f"mAP:\n{mAP.mean():.3f}")
            print(f"AuC:{auc.mean():.3f}", file=wp)
            print(f"AuC:\n{auc.mean():.3f}")


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    fire.Fire(Runner)
