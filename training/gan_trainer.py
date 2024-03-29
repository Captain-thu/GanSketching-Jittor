import os
import numpy as np
# import torch
# import torchvision.utils
import jittor as jt
from jittor import misc

from .gan_model import GANModel


class GANTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        # self.device = 'cuda' if not self.opt.use_cpu else 'cpu'

        self.gan_model = GANModel(opt)

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.gan_model.create_optimizers(opt)
            self.gan_model.create_loss_fns(opt)
            self.gan_model.set_requires_grad(False, False)

            if opt.resume_iter is not None:
                self.load(opt.resume_iter)

            self.g_losses = {}
            self.d_losses = {}
            self.interm_imgs = {}
            self.set_fixed_noise()

    def run_generator_one_step(self, data):
        g_losses, generated = self.gan_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        self.optimizer_G.zero_grad()
        self.optimizer_G.backward(g_loss)
        self.optimizer_G.step()
        self.generated = generated
        update_dict(self.g_losses, g_losses)

    def run_discriminator_one_step(self, data):
        d_losses, interm_imgs = self.gan_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        self.optimizer_D.zero_grad()
        self.optimizer_D.backward(d_loss)
        self.optimizer_D.step()
        update_dict(self.d_losses, d_losses)
        update_dict(self.interm_imgs, interm_imgs)
        return

    def run_discriminator_regularization_one_step(self, data):
        d_reg_losses = self.gan_model(data, mode='discriminator-regularize')
        d_reg_loss = sum(d_reg_losses.values()).mean()
        self.optimizer_D.zero_grad()
        self.optimizer_D.backward(d_reg_loss)
        self.optimizer_D.step()
        update_dict(self.d_losses, d_reg_losses)

    def train_one_step(self, data, iters):
        self.gan_model.set_requires_grad(False, True)
        print('\nrun discriminator one step')
        self.run_discriminator_one_step(data)

        if not self.opt.no_d_regularize and iters % self.opt.d_reg_every == 0:
            print('run discriminator regularization one step')
            self.run_discriminator_regularization_one_step(data)

        self.gan_model.set_requires_grad(True, False)
        print('run generator one step')
        self.run_generator_one_step(data)

    def get_latest_losses(self):
        self.reports = {**self.g_losses, **self.d_losses}
        return {k: v.mean().item() for k, v in self.reports.items()}

    def get_latest_generated(self):
        return self.generated

    def get_visuals(self):
        visuals = {}
        print("visuals:get samples")
        sample, transf = self.gan_model.inference(self.sample_z, with_tf=True)
        interp = self.gan_model.inference(self.interp_z)

        print("visuals:get samples trunc")
        sample_trunc = self.gan_model.inference(self.sample_z, trunc_psi=0.5)
        interp_trunc = self.gan_model.inference(self.interp_z, trunc_psi=0.5)

        print("visuals:calculate")
        visuals['sample'] = misc.make_grid(sample, nrow=8)
        visuals['sample_transf'] = misc.make_grid(transf, nrow=8)
        visuals['interp'] = misc.make_grid(interp, nrow=8)
        visuals['sample_psi0.5'] = misc.make_grid(sample_trunc, nrow=8)
        visuals['interp_psi0.5'] = misc.make_grid(interp_trunc, nrow=8)
        update_dict(visuals, self.interm_imgs)
        return visuals

    def get_gan_model(self):
        return self.gan_model

    def save(self, iters):
        self.gan_model.save(iters)

        misc = {
            "g_optim": self.optimizer_G.state_dict(),
            "d_optim": self.optimizer_D.state_dict(),
            "opt": self.opt,
        }
        save_path = os.path.join(
            self.opt.checkpoints_dir, self.opt.name, f"{iters}_net_")
        jt.save(misc, save_path + "misc.pth")

    def load(self, iters):
        print(f"Resuming model at iteration {iters}")
        self.gan_model.load(iters)
        load_path = os.path.join(
            self.opt.checkpoints_dir, self.opt.name, f"{iters}_net_")
        # state_dict = jt.load(load_path + "misc.pth", map_location=self.device)
        state_dict = jt.load(load_path + "misc.pth")
        self.optimizer_G.load_state_dict(state_dict["g_optim"])
        self.optimizer_D.load_state_dict(state_dict["d_optim"])

    def set_fixed_noise(self):
        os.makedirs('./cache_files/', exist_ok=True)
        if self.opt.reduce_visuals:
            sample_z_file = './cache_files/sample_z_reduced.pth'
            interp_z_file = './cache_files/interp_z_reduced.pth'
        else:
            sample_z_file = './cache_files/sample_z.pth'
            interp_z_file = './cache_files/interp_z.pth'

        if os.path.exists(sample_z_file):
            self.sample_z = jt.load(sample_z_file)
        else:
            if self.opt.reduce_visuals:
                z = jt.randn(8, 512)
            else:
                z = jt.randn(32, 512)

            jt.save(z, sample_z_file)
            self.sample_z = z

        if os.path.exists(interp_z_file):
            self.interp_z = jt.load(interp_z_file)
        else:
            with jt.no_grad():
                if self.opt.reduce_visuals:
                    z0 = jt.randn(1, 1, 512)
                    z1 = jt.randn(1, 1, 512)
                else:
                    z0 = jt.randn(4, 1, 512)
                    z1 = jt.randn(4, 1, 512)

                z = []
                for c in np.linspace(0, 1, 8):
                    z.append((1 - c) * z0 + c * z1)
                z = jt.concat(z, 1).view(-1, 512)
            jt.save(z, interp_z_file)
            self.interp_z = z


def update_dict(old_dict, new_dict):
    for key in new_dict.keys():
        old_dict[key] = new_dict[key]
