"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./dataset --dataset_mode rain --model raincycle --name JRGR
"""
import os
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.            
    lst_models = []
    lst_model_numbers = []
    lst_epoch_numbers = []
    n_models = opt.n_models
    model_epochs = opt.model_epochs
    arr_epochs = model_epochs.split(',')
    
    for ep in arr_epochs:
        opt.epoch = ep
        for j in range(1, n_models + 1, 1):
                print("model #:", j)
                opt.modelnumber = j
                model = create_model(opt)      # create a model given opt.model and other options
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                print("model Created")
                if opt.eval:
                    model.eval()
                    
                lst_model_numbers.append(j)
                lst_epoch_numbers.append(ep)
                lst_models.append(model)
    
    zip_models = zip(lst_epoch_numbers, lst_model_numbers, lst_models)
    epoch_model_wise_evals = {}
    epoch_model_combination_wise_evals = {}
    epoch_wise_evals = {}
    
    for i, data in enumerate(dataset):
        lst_imobj = {}
        for rr_model in lst_models:
            model = rr_model                # create a model given opt.model and other options
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            print("model Visuals created")
            for label, im_data in visuals.items():
                if label in lst_imobj:
                    arr_imgs = lst_imobj[label]
                    arr_imgs.append(im_data)                    
                else:
                    arr_imgs = []
                    arr_imgs.append(im_data)
                
                lst_imobj[label] = arr_imgs
            print("model Visuals collected")
            
        lable == "pred_Bt":
        Bt_img = lst_imobj["Bt"][0]
        arr_imdata = lst_imobj[lable]
        
        for ep, mn, model in zip_models:                        
            for pred_bt_img in arr_imdata:
                psnr = util.calculate_PSNR(Bt_img, pred_bt_img)
                ssim = util.calculate_ssim(Bt_img, pred_bt_img)
                
                if ep in epoch_model_wise_evals:
                    model_wise_evals = epoch_model_wise_evals[ep]
                    
                    if mn in model_wise_evals:
                        evals = model_wise_evals[mn]
                        
                        if "psnr" in evals:
                            arr_psnr = evals["psnr"]
                            arr_psnr.append(psnr)
                            evals["psnr"] = arr_psnr
                        else:
                            arr_psnr = [psnr]
                            evals["psnr"] = arr_psnr
                        
                        if "ssim" in evals:
                            arr_ssim = evals["ssim"]
                            arr_ssim.append(ssim)
                            evals["ssim"] = arr_ssim
                        else:
                            arr_ssim = [ssim]
                            evals["ssim"] = arr_ssim
                    else:
                        evals = {}
                        arr_psnr = [psnr]
                        arr_ssim = [ssim]
                        evals["psnr"] = arr_psnr
                        evals["ssim"] = arr_ssim
                        model_wise_evals[mn] = evals
                
                else:
                    model_wise_evals = {}
                    evals = {}
                    arr_psnr = [psnr]
                    arr_ssim = [ssim]
                    evals["psnr"] = arr_psnr
                    evals["ssim"] = arr_ssim
                    model_wise_evals[mn] = evals 
                    epoch_model_wise_evals[ep] = model_wise_evals
            
        k=0
        for ep in arr_epochs:
            for i in range(0, n_models):
                for j in range(i+1, n_models):
                    mn = str(i) + "-" + str(j)
                    pred_bt_img = arr_imdata[i:j]

                    pred_bt_img_mean = torch.mean(torch.stack(pred_bt_img), dim=0)
                    pred_bt_img_min = torch.min(torch.stack(pred_bt_img), dim=0).values
                    pred_bt_img_max = torch.max(torch.stack(pred_bt_img), dim=0).values

                    psnr_mean = util.calculate_PSNR(Bt_img, pred_bt_img_mean)
                    ssim_mean = util.calculate_ssim(Bt_img, pred_bt_img_mean)

                    psnr_min = util.calculate_PSNR(Bt_img, pred_bt_img_mean)
                    ssim_min = util.calculate_ssim(Bt_img, pred_bt_img_mean)

                    psnr_max = util.calculate_PSNR(Bt_img, pred_bt_img_mean)
                    ssim_max = util.calculate_ssim(Bt_img, pred_bt_img_mean)

                    if ep in epoch_model_combination_wise_evals:
                        model_wise_evals = epoch_model_combination_wise_evals[ep]

                        if mn in model_wise_evals:
                            evals = model_wise_evals[mn]

                            if "psnr_mean" in evals:
                                arr_psnr = evals["psnr_mean"]
                                arr_psnr.append(psnr_mean)
                                evals["psnr_mean"] = arr_psnr
                            else:
                                arr_psnr = [psnr_mean]
                                evals["psnr_mean"] = arr_psnr

                            if "ssim_mean" in evals:
                                arr_ssim = evals["ssim_mean"]
                                arr_ssim.append(ssim_mean)
                                evals["ssim_mean"] = arr_ssim
                            else:
                                arr_ssim = [ssim_mean]
                                evals["ssim_mean"] = arr_ssim

                            if "psnr_min" in evals:
                                arr_psnr = evals["psnr_min"]
                                arr_psnr.append(psnr_min)
                                evals["psnr_min"] = arr_psnr
                            else:
                                arr_psnr = [psnr_min]
                                evals["psnr_min"] = arr_psnr

                            if "ssim_min" in evals:
                                arr_ssim = evals["ssim_min"]
                                arr_ssim.append(ssim_min)
                                evals["ssim_min"] = arr_ssim
                            else:
                                arr_ssim = [ssim_min]
                                evals["ssim_min"] = arr_ssim

                            if "psnr_max" in evals:
                                arr_psnr = evals["psnr_max"]
                                arr_psnr.append(psnr_max)
                                evals["psnr_max"] = arr_psnr
                            else:
                                arr_psnr = [psnr_max]
                                evals["psnr_max"] = arr_psnr

                            if "ssim_max" in evals:
                                arr_ssim = evals["ssim_max"]
                                arr_ssim.append(ssim_max)
                                evals["ssim_max"] = arr_ssim
                            else:
                                arr_ssim = [ssim_max]
                                evals["ssim_max"] = arr_ssim
                        else:
                            evals = {}

                            arr_psnr = [psnr_mean]
                            arr_ssim = [ssim_mean]
                            evals["psnr_mean"] = arr_psnr
                            evals["ssim_mean"] = arr_ssim

                            arr_psnr = [psnr_min]
                            arr_ssim = [ssim_min]
                            evals["psnr_min"] = arr_psnr
                            evals["ssim_min"] = arr_ssim

                            arr_psnr = [psnr_max]
                            arr_ssim = [ssim_max]
                            evals["psnr_max"] = arr_psnr
                            evals["ssim_max"] = arr_ssim

                            model_wise_evals[mn] = evals

                    else:
                        model_wise_evals = {}
                        evals = {}
                        arr_psnr = [psnr_mean]
                        arr_ssim = [ssim_mean]
                        evals["psnr_mean"] = arr_psnr
                        evals["ssim_mean"] = arr_ssim

                        arr_psnr = [psnr_min]
                        arr_ssim = [ssim_min]
                        evals["psnr_min"] = arr_psnr
                        evals["ssim_min"] = arr_ssim

                        arr_psnr = [psnr_max]
                        arr_ssim = [ssim_max]
                        evals["psnr_max"] = arr_psnr
                        evals["ssim_max"] = arr_ssim

                        model_wise_evals[mn] = evals 
                        epoch_model_combination_wise_evals[ep] = model_wise_evals
            
            length = n_models
            ep_pred_bt_img = arr_imdata[k:length]
            k = k + n_models
            length = length + n_models
            
            pred_bt_img_mean = torch.mean(torch.stack(ep_pred_bt_img), dim=0)
            pred_bt_img_min = torch.min(torch.stack(ep_pred_bt_img), dim=0).values
            pred_bt_img_max = torch.max(torch.stack(ep_pred_bt_img), dim=0).values

            psnr_mean = util.calculate_PSNR(Bt_img, pred_bt_img_mean)
            ssim_mean = util.calculate_ssim(Bt_img, pred_bt_img_mean)

            psnr_min = util.calculate_PSNR(Bt_img, pred_bt_img_mean)
            ssim_min = util.calculate_ssim(Bt_img, pred_bt_img_mean)

            psnr_max = util.calculate_PSNR(Bt_img, pred_bt_img_mean)
            ssim_max = util.calculate_ssim(Bt_img, pred_bt_img_mean)
            
            if ep in epoch_wise_evals:
                evals = epoch_wise_evals[ep]
                
                if "psnr_mean" in evals:
                    arr_psnr = evals["psnr_mean"]
                    arr_psnr.append(psnr_mean)
                    evals["psnr_mean"] = arr_psnr
                else:
                    arr_psnr = [psnr_mean]
                    evals["psnr_mean"] = arr_psnr

                if "ssim_mean" in evals:
                    arr_ssim = evals["ssim_mean"]
                    arr_ssim.append(ssim_mean)
                    evals["ssim_mean"] = arr_ssim
                else:
                    arr_ssim = [ssim_mean]
                    evals["ssim_mean"] = arr_ssim

                if "psnr_min" in evals:
                    arr_psnr = evals["psnr_min"]
                    arr_psnr.append(psnr_min)
                    evals["psnr_min"] = arr_psnr
                else:
                    arr_psnr = [psnr_min]
                    evals["psnr_min"] = arr_psnr

                if "ssim_min" in evals:
                    arr_ssim = evals["ssim_min"]
                    arr_ssim.append(ssim_min)
                    evals["ssim_min"] = arr_ssim
                else:
                    arr_ssim = [ssim_min]
                    evals["ssim_min"] = arr_ssim

                if "psnr_max" in evals:
                    arr_psnr = evals["psnr_max"]
                    arr_psnr.append(psnr_max)
                    evals["psnr_max"] = arr_psnr
                else:
                    arr_psnr = [psnr_max]
                    evals["psnr_max"] = arr_psnr

                if "ssim_max" in evals:
                    arr_ssim = evals["ssim_max"]
                    arr_ssim.append(ssim_max)
                    evals["ssim_max"] = arr_ssim
                else:
                    arr_ssim = [ssim_max]
                    evals["ssim_max"] = arr_ssim
            else:
                evals = {}

                arr_psnr = [psnr_mean]
                arr_ssim = [ssim_mean]
                evals["psnr_mean"] = arr_psnr
                evals["ssim_mean"] = arr_ssim

                arr_psnr = [psnr_min]
                arr_ssim = [ssim_min]
                evals["psnr_min"] = arr_psnr
                evals["ssim_min"] = arr_ssim

                arr_psnr = [psnr_max]
                arr_ssim = [ssim_max]
                evals["psnr_max"] = arr_psnr
                evals["ssim_max"] = arr_ssim

                epoch_wise_evals[ep] = evals

        mean = torch.mean(torch.stack(arr_imdata), dim=0)
        print("Mean Calculated")
        visuals[lable] = mean

        img_path = model.get_image_paths()     # get image paths        
        if i % 2 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
        
    with open('/content/JRGR/drive/MyDrive/Thesis1/JRGR/epoch_wise_evals.json', 'w') as filehandle:
              json.dump(epoch_wise_evals, filehandle)            
    with open('/content/JRGR/drive/MyDrive/Thesis1/JRGR/epoch_model_combination_wise_evals.json', 'w') as filehandle:
              json.dump(epoch_model_combination_wise_evals, filehandle)
    with open('/content/JRGR/drive/MyDrive/Thesis1/JRGR/epoch_model_wise_evals.json', 'w') as filehandle:
                  json.dump(epoch_model_wise_evals, filehandle)
