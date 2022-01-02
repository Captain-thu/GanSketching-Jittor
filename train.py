import time
# import torch
# import torch.multiprocessing as mp

from options import get_opt, print_options
from eval import Evaluator
from util.visualizer import Visualizer
from training.gan_trainer import GANTrainer
from training.dataset import create_dataloader, yield_data

import jittor as jt

if jt.has_cuda:
    jt.flags.use_cuda = 1

# jt.cudnn.set_max_workspace_ratio(0.0)
jt.flags.lazy_execution = 0


# 有一个list越来越大
def training_loop():
    # torch.backends.cudnn.benchmark = True
    opt, parser = get_opt()
    opt.isTrain = True

    # needs to switch to spawn mode to be compatible with evaluation

    # dataloader for user sketches
    print('Create dataloader')
    dataloader_sketch, _ = create_dataloader(opt.dataroot_sketch,
                                             opt.size,
                                             opt.batch,
                                             opt.sketch_channel)
    # dataloader for image regularization
    if opt.dataroot_image is not None:
        dataloader_image, sampler_image = create_dataloader(opt.dataroot_image,
                                                            opt.size,
                                                            opt.batch)
        data_yield_image = yield_data(dataloader_image, sampler_image)

    print('Create trainer')
    trainer = GANTrainer(opt)

    print_options(parser, opt)
    trainer.gan_model.print_trainable_params()
    print('Create evaluator')
    if not opt.disable_eval:
        evaluator = Evaluator(opt, trainer.get_gan_model())
    # create a visualizer that display/save images and plots
    print('Create visualizer')
    visualizer = Visualizer(opt)

    # the total number of training iterations
    if opt.resume_iter is None:
        total_iters = 0
    else:
        total_iters = opt.resume_iter

    optimize_time = 0.1
    data = {'sketch': 1, 'image': 1}
    for epoch in range(opt.max_epoch):
        iter_data_time = time.time()    # timer for data loading per iteration
        # the number of training iterations in current epoch, reset to 0 every epoch
        epoch_iter = 0

        # inner loop within one epoch
        for data_sketch in dataloader_sketch:
            if total_iters >= opt.max_iter:
                return
            # makes dictionary to store all inputs
            data['sketch'] = data_sketch
            if opt.dataroot_image is not None:
                data['image'] = next(data_yield_image)

            # # timer for data loading per iteration
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            # # timer for optimization per iteration
            optimize_start_time = time.time()
            print('train one step')
            trainer.train_one_step(data, total_iters)
            optimize_time = (time.time() - optimize_start_time) * \
                0.005 + 0.995 * optimize_time

            # print training losses and save logging information to the disk
            if total_iters % opt.print_freq == 0:
                print("get loss")
                losses = trainer.get_latest_losses()
                visualizer.print_current_errors(
                    epoch, total_iters, losses, optimize_time, t_data)
                visualizer.plot_current_errors(losses, total_iters)

            # display images on wandb and save images to a HTML file
            if total_iters % opt.display_freq == 0:
                print('get visuals')
                visuals = trainer.get_visuals()
                visualizer.display_current_results(visuals, epoch, total_iters)

            # cache our latest model every <save_latest_freq> iterations
            if total_iters % opt.save_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' %
                      (epoch, total_iters))
                # it's useful to occasionally show the experiment name on console
                print(opt.name)
                trainer.save(total_iters)

            # evaluate the latest model
            with jt.no_grad():
                if not opt.disable_eval and total_iters % opt.eval_freq == 0:
                    metrics_start_time = time.time()
                    print("evaluate")
                    metrics, _ = evaluator.run_metrics(total_iters)
                    metrics_time = time.time() - metrics_start_time

                    visualizer.print_current_metrics(
                        epoch, total_iters, metrics, metrics_time)
                    visualizer.plot_current_errors(metrics, total_iters)

            total_iters += 1
            epoch_iter += 1
            iter_data_time = time.time()
            jt.sync_all()
            jt.gc()
            jt.clean()


if __name__ == "__main__":
    training_loop()
    print('Training was successfully finished.')
