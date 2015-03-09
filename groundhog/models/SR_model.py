"""
Implementation of a language model class.


TODO: write more documentation
"""
from collections import OrderedDict
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy
import itertools
import logging
import re

import cPickle as pkl

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams

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
                  state=None,
                  rnnencdec=None,
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
        self.state = state
        self.rnnencdec = rnnencdec
    
        self.valid_costs = ['log_p_expl','log_p_word']
        
        for prop in self.cost_layer.properties:
            if prop[0][:4] == 'ali_':
                self.valid_costs.append(prop[0])
        
        # Assume a single cost
        # We need to merge these lists
        state_below = self.cost_layer.state_below
        if hasattr(self.cost_layer, 'mask') and self.cost_layer.mask:
            num_words = TT.sum(self.cost_layer.mask)
        else:
            num_words = TT.cast(state_below.shape[0], 'float32')
        
        #do the weight decay
        logger.info('Computing weight decay')
        matched_params = {}
        for p in self.params:
            for param_name_pattern, wd in state['weight_decay_rules']:
                if re.match(param_name_pattern, p.name):
                    if p in matched_params:
                        logger.warn('multiple weight decay rules match: %s', p.name)
                    logger.info('Decaying %s by %s', p.name, wd)
                    matched_params[p] = (p**2).sum() * wd
        if matched_params:
            wdec_cost = sum(matched_params.values())
        else:
            wdec_cost = None

        if wdec_cost is not None:
            if self.state['normalize_by_batch_size']==False:
                in_names = [inp.name for inp in self.output_layer.inputs]
                y = self.output_layer.inputs[in_names.index('y')]
                assert y.name=='y'
                if y.ndim==2:
                    wdec_cost = wdec_cost *  y.shape[1]
                batch_size =  y.shape[1]
            self.properties.append(('wdec_cost', wdec_cost))
            self.train_cost = self.train_cost + wdec_cost
            self.properties.append(('tot_cost', self.train_cost))
            wdec_grad = theano.grad(wdec_cost, self.params, disconnected_inputs='ignore')
            self.param_grads = [wg+mg for wg,mg in zip(wdec_grad, self.param_grads)]
        
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
        if self.state['normalize_by_batch_size']==False:
            new_properties.append(('bs', batch_size))
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
        self.valid_tot_batch_cost = None

    def censor_updates(self, updates):
        clipped_updates = []
        for u in updates:
            p, p_up = u
            new_up = p_up
            matched = False
            for param_name_pattern, param_clip in self.state['weight_column_norm_clip_rules']:
                if re.match(param_name_pattern, p.name) and not p.name.endswith('__ls2'):
                    if matched:
                        logger.warn('multiple column norm clip rules match: %s', p.name)
                    matched=True
                    logger.info('Clipping columns of %s to %s', p.name, param_clip)
                    if param_clip is None:
                        new_up = p_up
                    else:
                        p_col_norms = TT.sqrt(TT.sum(TT.sqr(p_up), axis=0))
                        desired_norms = TT.clip(p_col_norms, 0, param_clip)
                        new_up = p_up * (desired_norms / (1e-7 + p_col_norms))
            
            #todo make generic    
            if self.state.get('enc_freeze_approx_embdr') and re.match('.*enc_approx_emb.*', p.name):
                logger.info('Freezinf %s', p.name)
                new_up = p
                    
            clipped_updates.append((p,new_up))
        return clipped_updates

    def validate(self, data_iterator, train=False):
        data_iterator.reset()
        import gc
        if self.valid_step is None:
            logger.debug('Compiling validation funcion')
            if self.valid_tot_batch_cost is None:
                tot_batch_cost = self.cost_layer.cost_per_sample.sum()
                extra_costs = []
                for prop in self.cost_layer.properties:
                    if prop[0][:4] == 'ali_':
                        extra_costs.append(prop)
                
                self.valid_extra_cost_names =  [p[0] for p in extra_costs]
                costs = [tot_batch_cost] + [p[1].sum() for p in extra_costs]
                
                if hasattr(self.rnnencdec, 'clean_trans_x'):
                    #Dropout - replace trans_x with cleran_trans_x
                    trans_x = self.rnnencdec.trans_x
                    if hasattr(trans_x, 'out'):
                        trans_x = trans_x.out
                    clean_trans_x = self.rnnencdec.clean_trans_x
                    if hasattr(clean_trans_x, 'out'):
                        clean_trans_x = clean_trans_x.out
                    
                    costs = theano.clone(costs, 
                                                  replace={trans_x: clean_trans_x}, 
                                                  share_inputs=True)
                self.valid_tot_batch_cost = costs
                
                         
            self.valid_step = theano.function(inputs=self.inputs, 
                                              outputs=self.valid_tot_batch_cost, 
                                              no_default_updates=True
                                              )
        
        gc.collect()
        gc.collect()
        gc.collect()
        
        costs = [0.0] * len(self.valid_tot_batch_cost)
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
            batch_costs = self.valid_step( **vals)
            for i in xrange(len(batch_costs)):
                if i==0 or not self.state['normalize_by_batch_size']:
                    costs[i] += batch_costs[i]
                else:
                    costs[i] += batch_costs[i]*y_mask.shape[1]
                    
        #ugly hack to prevent out-of memory errors
        #self.valid_step = None
        print 'did %d utts with %d words' %(n_expls, n_words)
        gc.collect()
        gc.collect()
        gc.collect()
        
        ret = [('log_p_expl', costs[0] / n_expls ),
                ('log_p_word', costs[0] / n_words )
                ]
        
        for ec_name, ec_val in zip(self.valid_extra_cost_names, costs[1:]):
            ret.append((ec_name, ec_val/n_expls))
        
        return ret

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

    
    def add_variational_noise(self):
        init_sigma = self.state['init_sigma']
        
        N = numpy.float32(self.state['Num_train_examples'])
        model_cost_coeff = numpy.float32(self.state.get('model_cost', 1.0))
        assert N>0
        
        theano_rng = RandomStreams(self.state['seed'])
        
        P_noisy = self.params
        P_clean = []
        
        Beta = []
        P_with_noise = []
        for p in P_noisy:
            p_u = p
            p_val = p.get_value(borrow=True)
            p_ls2 = theano.shared((numpy.zeros_like(p_val) + numpy.log(init_sigma)/2.).astype(dtype=numpy.float32), name='%s__ls2' % (p.name,))
            p_s2 = TT.exp(p_ls2)
            Beta.append((p_u, p_ls2, p_s2))
            #p_noisy = theano_rng.normal(size=p_val.shape, avg=p_u, std=TT.sqrt(p_s2), dtype='float32')
            #p_noisy = p_u + TT.ones_like(p_u)*0.001
            p_noisy = p_u + theano_rng.normal(size=p_val.shape)*TT.exp(p_ls2/2.0)
            p_noisy = TT.patternbroadcast(p_noisy, p.type.broadcastable)
            P_with_noise.append(p_noisy)
        
        #compute the prior mean and variation
        temp_sum = 0.0
        temp_param_count = 0.0
        for p_u,unused_p_ls2,unused_p_s2 in Beta:
            temp_sum = temp_sum + p_u.sum()
            temp_param_count = temp_param_count + p_u.shape.prod()
            
        prior_u = TT.cast(temp_sum/temp_param_count, 'float32')
        
        temp_sum = 0.0
        for p_u,unused_ls2,p_s2 in Beta:
            temp_sum = temp_sum + (p_s2**2).sum() + (((p_u-prior_u)**2).sum())
        
        prior_s2 = TT.cast(temp_sum/temp_param_count, 'float32')
        
        #convert everything to use the noisy parameters
        assert not self.updates #what to do with this
        f_properties = [p for p in self.properties if p[0] != 'grad_norm']
        
        to_convert = [self.train_cost] + self.param_grads + [fp[1] for fp in f_properties]
        converted = theano.clone(to_convert, replace=zip(P_noisy, P_with_noise))
        
        LC = 0.0
        for p_u,unused_ls2,p_s2 in Beta:
            LC = LC + 0.5*TT.log(prior_s2 / p_s2).sum() + 1.0 / (2.0 * prior_s2) * (((p_u-prior_u)**2).sum() + p_s2.sum() - prior_s2)
            
        LC = LC/N * model_cost_coeff
        
        self.train_cost = converted[0] + LC
        
        converted_grads = converted[1:len(self.param_grads)+1]
        converted_props = zip([fp[0] for fp in f_properties], converted[len(self.param_grads)+1:])
        converted_grads_dict = dict(zip(self.params, converted_grads))
        
        new_params = list(P_clean)
        new_grads = [converted_grads_dict[p] for p in new_params]
        
        assert self.state['bs'] == 1 #we need \sum_bach dLdW**2, but the best we can easily 
                                     #get is (\sum_bach dLdW)**2. They are equal when bs==1
        for p_u,p_ls2,p_s2 in Beta:
            p_grad = converted_grads_dict[p_u]
            p_u_grad = model_cost_coeff * (p_u - prior_u) / (N*prior_s2) + p_grad
            p_s2_grad = numpy.float32(model_cost_coeff * 0.5/N) * (1.0/prior_s2 + 1.0/p_s2) + 0.5 * p_grad**2 #figure a fix for bs>1
            
            p_ls2_grad = p_s2 * p_s2_grad
          
            new_params.append(p_u)
            new_params.append(p_ls2)
            new_grads.append(p_u_grad)
            new_grads.append(p_ls2_grad)
        
        self.params = new_params
        self.param_grads = new_grads
        
        grad_norm = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(self.param_grads, self.params) if p not in
                self.exclude_params_for_norm))
        
        self.properties = [('grad_norm', grad_norm)] + converted_props
        self.properties.append(('LC', LC))
        self.properties.append(('prior_u',prior_u))
        self.properties.append(('prior_s2',prior_s2))
        
            

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


