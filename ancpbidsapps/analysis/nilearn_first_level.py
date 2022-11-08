import argparse
import itertools
import os
import re
import math
from warnings import warn

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Memory
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel
from scipy.stats import norm

import ancpbids
from ancpbidsapps.app import App

CONTRAST_REGEX = re.compile(r"\w+-\w+")


class NilearnFirstLevelApp(App):
    """
    First level analysis of a BIDS dataset using Nilearn facilities.
    """

    def get_args_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-dp', '--dataset_path', type=str, required=True,
                            help='the dataset path containing a BIDS compliant dataset')
        parser.add_argument('-dvf', '--derivatives_folder', type=str, required=True,
                            help='the fmriprep generated preprocessing folder')
        parser.add_argument('-tl', '--task_label', type=str, required=True, help='the task name of the experiment')
        parser.add_argument('-c', '--contrast', type=str, required=True,
                            help='two conditions to use as contrast, '
                                 'example: language-idle, where "languagee" is first condition '
                                 'and "idle" the second to contrast with')
        parser.add_argument('-if', '--img_filters', type=str, nargs='*',
                            help='a list of filters to limit to specific imaging files')
        return parser

    def execute(self, **args):
        dataset_path, task_label, img_filters, derivatives_folder = args['dataset_path'], args['task_label'], args[
            'img_filters'], args['derivatives_folder']
        contrast = args['contrast']
        # simulate itertools.pairwise()
        if img_filters:
            img_filters = list(zip(*(itertools.islice(img_filters, i, None) for i in range(2))))

        layout = ancpbids.BIDSLayout(dataset_path)
        schema = layout.schema

        # derive data for fitting
        models, models_run_imgs, models_events, models_confounds = self.first_level(
            layout=layout,
            task_label=task_label,
            derivatives_folder=derivatives_folder,
            img_filters=img_filters)
        p001_unc = norm.isf(0.001)
        nsubj = len(models)
        ncols = int(math.sqrt(nsubj))
        nrows = int(nsubj / ncols)
        nrows = nrows + (nsubj - ncols * nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 4.5))
        model_and_args = zip(models, models_run_imgs, models_events, models_confounds)

        derivative = layout.dataset.create_derivative(name="ancpbidsfla")
        derivative.dataset_description.GeneratedBy.Name = "AncpBIDS First Level Analysis Example Pipeline"

        for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
            subject = derivative.create_folder(type_=schema.Subject, name='sub-' + model.subject_label)

            # fit the GLM
            model.fit(imgs, events, confounds)

            # compute the contrast of interest
            zmap = model.compute_contrast(contrast)
            zmap_artifact = subject.create_artifact()
            zmap_artifact.add_entity('desc', "nilearn")
            zmap_artifact.add_entity('task', task_label)
            zmap_artifact.suffix = 'zmap'
            zmap_artifact.extension = ".nii.gz"
            zmap_artifact.content = lambda file_path, z=zmap: z.to_filename(file_path)

            plotting.plot_glass_brain(zmap, colorbar=False, threshold=p001_unc,
                                      title=subject.name,
                                      axes=axes[
                                          int(midx / ncols), int(midx % ncols)],
                                      plot_abs=False, display_mode='x')

            plot_artifact = subject.create_artifact()
            plot_artifact.add_entity('desc', "nilearn")
            plot_artifact.add_entity('task', task_label)
            plot_artifact.suffix = "zmap"
            plot_artifact.extension = ".png"
            plot_artifact.content = lambda file_path, z=zmap: plotting.plot_glass_brain(z, colorbar=False,
                                                                                threshold=p001_unc,
                                                                                title=subject.name,
                                                                                plot_abs=False, display_mode='x',
                                                                                output_file=file_path)
        # remove empty subplots
        for ax in axes.flat[nsubj:]:
            ax.remove()
        fig.suptitle('subjects z_map contrast %s (unc p<%f)' % (contrast, p001_unc))

        fig_artifact = derivative.create_artifact()
        fig_artifact.add_entity('desc', "nilearn")
        fig_artifact.add_entity('task', task_label)
        fig_artifact.suffix = "zmap"
        fig_artifact.extension = ".png"
        fig_artifact.content = lambda file_path: fig.savefig(fname=file_path)

        layout.write_derivative(derivative)

    def first_level(self, layout: ancpbids.BIDSLayout, task_label, derivatives_folder, space_label=None,
                    img_filters=None, t_r=None, slice_time_ref=0.,
                    hrf_model='glover', drift_model='cosine',
                    high_pass=.01, drift_order=1, fir_delays=[0],
                    min_onset=-24, mask_img=None,
                    target_affine=None, target_shape=None,
                    smoothing_fwhm=None, memory=Memory(None),
                    memory_level=1, standardize=False,
                    signal_scaling=0, noise_model='ar1',
                    verbose=0, n_jobs=1,
                    minimize_memory=True):
        """Create FirstLevelModel objects and fit arguments from a BIDS dataset.

        It t_r is not specified this function will attempt to load it from a
        bold.json file alongside slice_time_ref. Otherwise t_r and slice_time_ref
        are taken as given.

        Parameters
        ----------
        dataset_path : str
            Directory of the highest level folder of the BIDS dataset. Should
            contain subject folders and a derivatives folder.

        task_label : str
            Task_label as specified in the file names like _task-<task_label>_.

        space_label : str, optional
            Specifies the space label of the preprocessed bold.nii images.
            As they are specified in the file names like _space-<space_label>_.

        img_filters : list of tuples (str, str), optional
            Filters are of the form (field, label). Only one filter per field
            allowed. A file that does not match a filter will be discarded.
            Possible filters are 'acq', 'ce', 'dir', 'rec', 'run', 'echo', 'res',
            'den', and 'desc'. Filter examples would be ('desc', 'preproc'),
            ('dir', 'pa') and ('run', '10').

        derivatives_folder : str, optional
            derivatives and app folder path containing preprocessed files.
            Like "derivatives/FMRIPREP". Default="derivatives".

        All other parameters correspond to a `FirstLevelModel` object, which
        contains their documentation. The subject label of the model will be
        determined directly from the BIDS dataset.

        Returns
        -------
        models : list of `FirstLevelModel` objects
            Each FirstLevelModel object corresponds to a subject. All runs from
            different sessions are considered together for the same subject to run
            a fixed effects analysis on them.

        models_run_imgs : list of list of Niimg-like objects,
            Items for the FirstLevelModel fit function of their respective model.

        models_events : list of list of pandas DataFrames,
            Items for the FirstLevelModel fit function of their respective model.

        models_confounds : list of list of pandas DataFrames or None,
            Items for the FirstLevelModel fit function of their respective model.

        """
        # check arguments
        img_filters = img_filters if img_filters else []
        if not isinstance(task_label, str):
            raise TypeError('task_label must be a string, instead %s was given' %
                            type(task_label))
        if space_label is not None and not isinstance(space_label, str):
            raise TypeError('space_label must be a string, instead %s was given' %
                            type(space_label))
        if not isinstance(img_filters, list):
            raise TypeError('img_filters must be a list, instead %s was given' %
                            type(img_filters))
        for img_filter in img_filters:
            if (not isinstance(img_filter[0], str)
                    or not isinstance(img_filter[1], str)):
                raise TypeError('filters in img filters must be (str, str), '
                                'instead %s was given' % type(img_filter))
            if img_filter[0] not in ['acq', 'ce', 'dir', 'rec', 'run',
                                     'echo', 'desc', 'res', 'den',
                                     ]:
                raise ValueError(
                    "field %s is not a possible filter. Only "
                    "'acq', 'ce', 'dir', 'rec', 'run', 'echo', "
                    "'desc', 'res', 'den' are allowed." % img_filter[0])

        # check derivatives folder is present
        if not layout.dataset.derivatives:
            raise ValueError('derivatives folder does not exist in given dataset')

        schema = layout.schema

        # Get acq specs for models. RepetitionTime and SliceTimingReference.
        # Throw warning if no bold.json is found
        if t_r is not None:
            warn('RepetitionTime given in model_init as %d' % t_r)
            warn('slice_time_ref is %d percent of the repetition '
                 'time' % slice_time_ref)
        else:
            filters = {'task': task_label}
            for img_filter in img_filters:
                if img_filter[0] in ['acq', 'rec', 'run']:
                    filters[img_filters[0]] = img_filters[1]

            metadata = layout.get_metadata(scope=derivatives_folder, suffix='bold', **filters)
            # If we don't find the parameter information in the derivatives folder
            # we try to search in the raw data folder
            if not metadata:
                metadata = layout.get_metadata(scope='raw', suffix='bold', **filters)
            if not metadata:
                warn('No bold.json found in derivatives folder or '
                     'in dataset folder. t_r can not be inferred and will need to'
                     ' be set manually in the list of models, otherwise their fit'
                     ' will throw an exception')
            else:
                if 'RepetitionTime' in metadata:
                    t_r = float(metadata['RepetitionTime'])
                else:
                    warn('RepetitionTime not found in any metadata file within dataset. t_r can not be '
                         'inferred and will need to be set manually in the '
                         'list of models. Otherwise their fit will throw an '
                         ' exception')
                if 'SliceTimingRef' in metadata:
                    slice_time_ref = float(metadata['SliceTimingRef'])
                else:
                    warn('SliceTimingRef not found in any metadata file within dataset. It will be assumed'
                         ' that the slice timing reference is 0.0 percent of the '
                         'repetition time. If it is not the case it will need to '
                         'be set manually in the generated list of models')

        # Get subjects in dataset
        sub_labels = layout.get_subjects()

        # Build fit_kwargs dictionaries to pass to their respective models fit
        # Events and confounds files must match number of imgs (runs)
        models = []
        models_run_imgs = []
        models_events = []
        models_confounds = []
        for sub_label in sub_labels:
            # Create model
            model = FirstLevelModel(
                t_r=t_r, slice_time_ref=slice_time_ref, hrf_model=hrf_model,
                drift_model=drift_model, high_pass=high_pass,
                drift_order=drift_order, fir_delays=fir_delays,
                min_onset=min_onset, mask_img=mask_img,
                target_affine=target_affine, target_shape=target_shape,
                smoothing_fwhm=smoothing_fwhm, memory=memory,
                memory_level=memory_level, standardize=standardize,
                signal_scaling=signal_scaling, noise_model=noise_model,
                verbose=verbose, n_jobs=n_jobs,
                minimize_memory=minimize_memory, subject_label=sub_label)
            models.append(model)

            # Get preprocessed imgs
            if space_label is None:
                filters = [('task', task_label)] + img_filters
            else:
                filters = [('task', task_label),
                           ('space', space_label)] + img_filters
            filters = {i[0]: i[1] for i in filters}
            imgs = layout.get(scope=derivatives_folder, suffix='bold', extension=['.nii', '.nii.gz'],
                              subject=sub_label, **filters)
            # If there is more than one file for the same (ses, run), likely we
            # have an issue of underspecification of filters.
            run_check_list = []
            # If more than one run is present the run field is mandatory in BIDS
            # as well as the ses field if more than one session is present.
            if len(imgs) > 1:
                for img in imgs:
                    if (
                            img.has_entity('ses')
                            and img.has_entity('run')
                    ):
                        entity = (
                            img.get_entity('ses'),
                            img.get_entity('run'))
                        if entity in run_check_list:
                            raise ValueError(
                                'More than one nifti image found '
                                'for the same run %s and session %s. '
                                'Please verify that the '
                                'desc_label and space_label labels '
                                'corresponding to the BIDS spec '
                                'were correctly specified.' % entity)
                        else:
                            run_check_list.append(entity)

                    elif img.has_entity('ses'):
                        if img.get_entity('ses') in run_check_list:
                            raise ValueError(
                                'More than one nifti image '
                                'found for the same ses %s, while '
                                'no additional run specification present'
                                '. Please verify that the desc_label and '
                                'space_label labels '
                                'corresponding to the BIDS spec '
                                'were correctly specified.' %
                                img.get_entity('ses'))
                        else:
                            run_check_list.append(img.get_entity('ses'))

                    elif img.has_entity('run'):
                        if img.get_entity('run') in run_check_list:
                            raise ValueError(
                                'More than one nifti image '
                                'found for the same run %s. '
                                'Please verify that the desc_label and '
                                'space_label labels '
                                'corresponding to the BIDS spec '
                                'were correctly specified.' %
                                img.get_entity('run'))
                        else:
                            run_check_list.append(img.get_entity('run'))
            img_paths = list(map(lambda a: a.get_absolute_path(), imgs))
            models_run_imgs.append(img_paths)

            # Get events and extra confounds
            filters = [('task', task_label)]
            for img_filter in img_filters:
                if img_filter[0] in ['acq', 'rec', 'run']:
                    filters.append(img_filter)
            # Get events files
            filters = {i[0]: i[1] for i in filters}
            events = layout.get(return_type='filenames', suffix='events', extension='.tsv',
                                subject=sub_label, **filters)
            if events:
                if len(events) != len(imgs):
                    raise ValueError('%d events.tsv files found for %d bold '
                                     'files. Same number of event files as '
                                     'the number of runs is expected' %
                                     (len(events), len(imgs)))
                events = [pd.read_csv(event, sep='\t', index_col=None)
                          for event in events]
                models_events.append(events)
            else:
                raise ValueError('No events.tsv files found')

            # Get confounds. If not found it will be assumed there are none.
            # If there are confounds, they are assumed to be present for all runs.
            confounds = layout.get(scope=derivatives_folder, return_type='filenames', extension='.tsv',
                                   desc='confounds', subject=sub_label, **filters)

            if confounds:
                if len(confounds) != len(imgs):
                    raise ValueError('%d confounds.tsv files found for %d bold '
                                     'files. Same number of confound files as '
                                     'the number of runs is expected' %
                                     (len(events), len(imgs)))
                confounds = [pd.read_csv(c, sep='\t', index_col=None)
                             for c in confounds]
                models_confounds.append(confounds)

        return models, models_run_imgs, models_events, models_confounds
