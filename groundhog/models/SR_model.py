"""
Implementation of a language model class.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy
import itertools
import logging

import cPickle as pkl

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import id_generator
from groundhog.layers.basic import Model

logger = logging.getLogger(__name__)

class SR_Model(Model):
    def  __init__(self,
                  cost_layer = None,
                  sample_fn = None,
                  valid_fn = None,
                  noise_fn = None,
                  clean_before_noise_fn = False,
                  clean_noise_validation=True,
                  weight_noise_amount = 0,
                  indx_word="/data/lisa/data/PennTreebankCorpus/dictionaries.npz",
                  need_inputs_for_generating_noise=False,
                  indx_word_src=None,
                  character_level = False,
                  exclude_params_for_norm=None,
                  rng = None):
        """
        Constructs a model, that respects the interface required by the
        trainer class.

        :type cost_layer: groundhog layer
        :param cost_layer: the cost (last) layer of the model

        :type sample_fn: function or None
        :param sample_fn: function used to sample from the model

        :type valid_fn: function or None
        :param valid_fn: function used to compute the validation error on a
            minibatch of examples

        :type noise_fn: function or None
        :param noise_fn: function called to corrupt an input (that
            potentially will be denoised by the model)

        :type clean_before_noise_fn: bool
        :param clean_before_noise_fn: If the weight noise should be removed
            before calling the `noise_fn` to corrupt some input

        :type clean_noise_validation: bool
        :param clean_noise_validation: If the weight noise should be removed
            before calling the validation function

        :type weight_noise_amount: float or theano scalar
        :param weight_noise_amount: weight noise scale (standard deviation
            of the Gaussian from which it is sampled)

        :type indx_word: string or None
        :param indx_word: path to the file describing how to match indices
            to words (or characters)

        :type need_inputs_for_generating_noise: bool
        :param need_inputs_for_generating_noise: flag saying if the shape of
            the inputs affect the shape of the weight noise that is generated at
            each step

        :type indx_word_src: string or None
        :param indx_word_src: similar to indx_word (but for the source
            language

        :type character_level: bool
        :param character_level: flag used when sampling, saying if we are
            running the model on characters or words

        :type excluding_params_for_norm: None or list of theano variables
        :param excluding_params_for_norm: list of parameters that should not
            be included when we compute the norm of the gradient (for norm
            clipping). Usually the output weights if the output layer is
            large

        :type rng: numpy random generator
        :param rng: numpy random generator

        """
        super(SR_Model, self).__init__(output_layer=cost_layer,
                                       sample_fn=sample_fn,
                                       indx_word=indx_word,
                                       indx_word_src=indx_word_src,
                                       rng=rng)
        if exclude_params_for_norm is None:
            self.exclude_params_for_norm = []
        else:
            self.exclude_params_for_norm = exclude_params_for_norm
        self.need_inputs_for_generating_noise=need_inputs_for_generating_noise
        self.cost_layer = cost_layer
        self.clean_noise_validation = clean_noise_validation
        self.noise_fn = noise_fn
        self.clean_before = clean_before_noise_fn
        self.weight_noise_amount = weight_noise_amount
        self.character_level = character_level

        self.valid_costs = ['log_p_expl','log_p_word']
        # Assume a single cost
        # We need to merge these lists
        state_below = self.cost_layer.state_below
        if hasattr(self.cost_layer, 'mask') and self.cost_layer.mask:
            num_words = TT.sum(self.cost_layer.mask)
        else:
            num_words = TT.cast(state_below.shape[0], 'float32')
        
        grad_norm = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(self.param_grads, self.params) if p not in
                self.exclude_params_for_norm))
        
        #note: cost per sample doesn't use scale!
        per_word_cost = self.cost_layer.cost_per_sample.sum() / num_words
        per_expl_cost = self.cost_layer.cost_per_sample.mean() 
        new_properties = [
                ('grad_norm', grad_norm),
                ('log_p_word', per_word_cost ),
                ('log_p_expl', per_expl_cost ),          
                ('log2_p_word', per_word_cost / numpy.float32(numpy.log(2))),
                ('log2_p_expl', per_expl_cost / numpy.float32(numpy.log(2)))]
        self.properties += new_properties

        if len(self.noise_params) >0 and weight_noise_amount:
            if self.need_inputs_for_generating_noise:
                inps = self.inputs
            else:
                inps = []
            self.add_noise = theano.function(inps,[],
                                             name='add_noise',
                                             updates = [(p,
                                                 self.trng.normal(shp_fn(self.inputs),
                                                     avg =0,
                                                     std=weight_noise_amount,
                                                     dtype=p.dtype))
                                                 for p, shp_fn in
                                                        zip(self.noise_params,
                                                         self.noise_params_shape_fn)],
                                            on_unused_input='ignore')
            self.del_noise = theano.function(inps,[],
                                             name='del_noise',
                                             updates=[(p,
                                                       TT.zeros(shp_fn(self.inputs),
                                                                p.dtype))
                                                      for p, shp_fn in
                                                      zip(self.noise_params,
                                                          self.noise_params_shape_fn)],
                                            on_unused_input='ignore')
        else:
            self.add_noise = None
            self.del_noise = None
        
        self.valid_step = None

    def validate(self, data_iterator, train=False):
        data_iterator.reset()
        
        if self.valid_step is None:
            logger.debug('Compiling validation funcion')
            tot_batch_cost = self.cost_layer.cost_per_sample.sum() 
            self.valid_step = theano.function(inputs=self.inputs, 
                                              outputs=tot_batch_cost, 
                                              no_default_updates=True
                                              )
        cost = 0
        n_expls = 0
        n_words = 0
                
        for vals in data_iterator:
            assert isinstance(vals, dict)
            if self.del_noise and self.clean_noise_validation:
                if self.need_inputs_for_generating_noise:
                    self.del_noise(**vals)
                else:
                    self.del_noise()

            y_mask = vals['y_mask']
            n_expls += y_mask.shape[1]
            n_words += y_mask.sum()
            cost += self.valid_step( **vals)
            
        return [('log_p_expl', cost / n_expls ),
                ('log_p_word', cost / n_words )
                ]

    def load_dict(self, opts):
        """
        Loading the dictionary that goes from indices to actual words
        """

        if self.indx_word and '.pkl' in self.indx_word[-4:]:
            data_dict = pkl.load(open(self.indx_word, "r"))
            self.word_indxs = data_dict
            self.word_indxs[opts['null_sym_target']] = '<eol>'
            self.word_indxs[opts['unk_sym_target']] = opts['oov']
        elif self.indx_word and '.np' in self.indx_word[-4:]:
            self.word_indxs = numpy.load(self.indx_word)['unique_words']

        if self.indx_word_src and '.pkl' in self.indx_word_src[-4:]:
            data_dict = pkl.load(open(self.indx_word_src, "r"))
            self.word_indxs_src = data_dict
            self.word_indxs_src[opts['null_sym_source']] = '<eol>'
            self.word_indxs_src[opts['unk_sym_source']] = opts['oov']
        elif self.indx_word_src and '.np' in self.indx_word_src[-4:]:
            self.word_indxs_src = numpy.load(self.indx_word_src)['unique_words']



    def get_samples(self, length = 30, temp=1, *inps):
        if not hasattr(self, 'word_indxs'):
           self.load_dict()
        self._get_samples(self, length, temp, *inps)

    def perturb(self, *args, **kwargs):
        if args:
            inps = args
            assert not kwargs
        if kwargs:
            inps = kwargs
            assert not args

        if self.noise_fn:
            if self.clean_before and self.del_noise:
                if self.need_inputs_for_generating_noise:
                    self.del_noise(*args, **kwargs)
                else:
                    self.del_noise()
            inps = self.noise_fn(*args, **kwargs)
        if self.add_noise:
            if self.need_inputs_for_generating_noise:
                self.add_noise(*args, **kwargs)
            else:
                self.add_noise()
        return inps


