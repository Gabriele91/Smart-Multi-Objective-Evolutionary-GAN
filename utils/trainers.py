import lasagne
import theano
import theano.tensor as T


class GeneratorTrainer:
    def __init__(self, noise, generator, discriminator, lr, b1):
        self.noise=noise
        self.generator=generator
        self.discriminator=discriminator
        self.Tgimgs = lasagne.layers.get_output(generator)
        self.Tfake_out = lasagne.layers.get_output(discriminator, self.Tgimgs)
        self.generator_params = lasagne.layers.get_all_params(generator, trainable=True)
        self.g_loss_logD = lasagne.objectives.binary_crossentropy(self.Tfake_out, 1).mean()
        self.g_loss_minimax = -lasagne.objectives.binary_crossentropy(self.Tfake_out, 0).mean()
        self.g_loss_ls = T.mean(T.sqr((self.Tfake_out - 1)))
        self.up_g_logD = lasagne.updates.adam(self.g_loss_logD, self.generator_params, learning_rate=lr, beta1=b1)
        self.up_g_minimax = lasagne.updates.adam(self.g_loss_minimax, self.generator_params, learning_rate=lr, beta1=b1)
        self.up_g_ls = lasagne.updates.adam(self.g_loss_ls, self.generator_params, learning_rate=lr, beta1=b1)
        self.train_g = theano.function([noise], self.g_loss_logD, updates=self.up_g_logD)
        self.train_g_minimax = theano.function([noise], self.g_loss_minimax, updates=self.up_g_minimax)
        self.train_g_ls = theano.function([noise], self.g_loss_ls, updates=self.up_g_ls)
        self.gen_fn = theano.function([noise],  lasagne.layers.get_output(generator, deterministic=True))
    
    def train(self,loss_type,zmb):
        if loss_type == 'trickLogD':
            return self.train_g(zmb)
        elif loss_type == 'minimax':
            return self.train_g_minimax(zmb)
        elif loss_type == 'ls':
            return self.train_g_ls(zmb)
        else:
            raise "{} is invalid loss".format(loss_type)

    def gen(self,zmb):
        return self.gen_fn(zmb)

    def set(self,params):
        lasagne.layers.set_all_param_values(self.generator, params)
        return self

    def get(self):
        return lasagne.layers.get_all_param_values(self.generator)

