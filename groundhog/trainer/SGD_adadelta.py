"""
Stochastic Gradient Descent.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"

import numpy
import time
import logging

import re

import theano
import theano.tensor as TT
from theano.sandbox.scan import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import print_time, print_mem, const

logger = logging.getLogger(__name__)

class SGD(object):
    def __init__(self,
                 model,
                 state,
                 data):
        """
        Parameters:
            :param model:
                Class describing the model used. It should provide the
                 computational graph to evaluate the model, and have a
                 similar structure to classes on the models folder
            :param state:
                Dictionary containing the current state of your job. This
                includes configuration of the job, specifically the seed,
                the startign damping factor, batch size, etc. See main.py
                for details
            :param data:
                Class describing the dataset used by the model
        """

        if 'adarho' not in state:
            state['adarho'] = 0.96
        if 'adaeps' not in state:
            state['adaeps'] = 1e-6
        if 'profile' not in state:
            state['profile'] = 0
        
        #####################################
        # Step 0. Constructs shared variables
        #####################################
        bs = state['bs']
        self.model = model
        self.rng = numpy.random.RandomState(state['seed'])
        srng = RandomStreams(self.rng.randint(213))
        self.gs = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name)
                   for p in model.params]
        self.gnorm2 = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_g2')
                   for p in model.params]
        self.dnorm2 = [theano.shared(numpy.zeros(p.get_value(borrow=True).shape,
                                             dtype=theano.config.floatX),
                                name=p.name+'_d2')
                   for p in model.params]

        self.step = 0
        self.bs = bs
        self.state = state
        self.data = data
        self.step_timer = time.time()
        self.gdata = [theano.shared(numpy.zeros( (2,)*x.ndim,
                                                dtype=x.dtype),
                                    name=x.name) for x in model.inputs]

        self.lr = theano.shared(numpy.float32(state['lr']), name='lr')

        if state['cutoff_adapt']:
            #start these at zero, since the initial cutoff is 1 and scaled gradient norm is max 1
            self.gnorm_log_ave = theano.shared( numpy.float32(0), name='gnorm_log_ave')
            #nasty init trick to have a low stdev at the beginning
            self.gnorm_log2_ave = theano.shared( numpy.float32(0), name='gnorm_log2_ave' )
            self.gnorm_log2_ave_gater = theano.shared( numpy.float32(0), name='gnorm_log2_ave_gater' )
            self.cutoff = theano.shared( numpy.float32(state['cutoff'], name='cutoff') )
            logger.debug("Will use adaptive cutoff computation")
        else:
            self.cutoff = theano.shared( numpy.float32(state['cutoff'], name='cutoff') )
            

        ###################################
        # Step 1. Compile training function
        ###################################
        logger.debug('Constructing grad function')
        loc_data = self.gdata
        
        #do the weight decay
        logger.info('Computing weight decay')
        wdec_cost = None
        matched_params = set()
        for p in model.params:
            for param_name_pattern, wd in state['weight_decay_rules']:
                if re.match(param_name_pattern, p.name):
                    if p in matched_params:
                        logger.warn('multiple weight decay rules match: %s', p.name)
                    matched_params.add(p)
                    logger.info('Decaying %s by %s', p.name, wd)
                    if wdec_cost is None:
                        wdec_cost =  (p**2).sum() * wd
                    else:
                        wdec_cost = wdec_cost + (p**2).sum() * wd

        if wdec_cost is not None:
            wdec_grad = theano.grad(wdec_cost, model.params, disconnected_inputs='ignore')
            tot_grad = [wg+mg for wg,mg in zip(wdec_grad, model.param_grads)]
            model.properties.append(('wdec_cost', wdec_cost))
        else:
            tot_grad = model.param_grads
                
        scaled_grads = [g*self.lr for g in tot_grad]
        
        norm_gs = TT.sqrt(sum(TT.sum(x**2)
                for x,p in zip(scaled_grads, model.params) if p not in model.exclude_params_for_norm))        
        norm_gs.name='scaled_grad_norm'
        
        model.properties.append(('scaled_grad_norm', norm_gs))
        
        self.prop_exprs = [x[1] for x in model.properties]
        self.prop_names = [x[0] for x in model.properties]
        self.update_rules = [x[1] for x in model.updates]
        rval = theano.clone(scaled_grads + self.update_rules + \
                            self.prop_exprs + [model.train_cost],
                            replace=zip(model.inputs, loc_data))
        nparams = len(model.params)
        nouts = len(self.prop_exprs)
        nrules = len(self.update_rules)
        gs = rval[:nparams]
        rules = rval[nparams:nparams + nrules]
        outs = rval[nparams + nrules:]
        
        norm_gs = outs[-2] #outs[-1] is cost, outs[-2] is the last model property, or scaled_grad_norm
        if 'cutoff' in state and state['cutoff'] > 0:
            c = self.cutoff
            if state['cutoff_rescale_length']:
                c = c * TT.cast(loc_data[0].shape[0], 'float32')

            notfinite = TT.or_(TT.isnan(norm_gs), TT.isinf(norm_gs))
            _gs = []
            for g,p in zip(gs,self.model.params):
                if p not in self.model.exclude_params_for_norm:
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*c/norm_gs, g)
                    _gs.append(
                       TT.switch(notfinite, numpy.float32(.1)*p, tmpg))
                       #TT.switch(notfinite, numpy.float32(0.0)*p, tmpg))
                else:
                    _gs.append(g)
            gs = _gs
        store_gs = [(s,g) for s,g in zip(self.gs, gs)]
        updates = store_gs + [(s[0], r) for s,r in zip(model.updates, rules)]

        rho = self.state['adarho']
        eps = self.state['adaeps']
        
        if state['cutoff_adapt']:
            cut_rho = self.state['cutoff_rho']
            gnorm_log = TT.log(norm_gs)
            gnorm_log_ave_up = (cut_rho*self.gnorm_log_ave + 
                                numpy.float32(1.-cut_rho) * gnorm_log)
            gnorm_log2_ave_up = (cut_rho*self.gnorm_log2_ave + 
                                 numpy.float32(1.-cut_rho) * gnorm_log**2)         
            cutoff_up = TT.exp(gnorm_log_ave_up +
                               self.gnorm_log2_ave_gater * 
                               TT.sqrt(TT.maximum(numpy.float32(0.0),
                                                  gnorm_log2_ave_up - gnorm_log_ave_up**2
                                              )
                                   ) * numpy.float32(state['cutoff_stdevs'])  
                           )
            #the idea is that only when we have a decent estimate of the stdew the gater will be 1 as they have the same decay constant
            gnorm_log2_ave_gater_up = TT.cast((cut_rho*self.gnorm_log2_ave_gater + (1.-cut_rho)*numpy.float32(1.0)), theano.config.floatX)
            if 1:
                cutoff_up = TT.minimum(numpy.float32(state['cutoff_max']),
                                       (cut_rho*self.cutoff + (1.-cut_rho) * cutoff_up
                                        ))
                     
            updates  = updates + [(self.gnorm_log_ave, TT.switch(notfinite, self.gnorm_log_ave, gnorm_log_ave_up)),
                                  (self.gnorm_log2_ave, TT.switch(notfinite, self.gnorm_log2_ave, gnorm_log2_ave_up)),
                                  (self.cutoff, TT.switch(notfinite, self.cutoff, cutoff_up)),
                                  (self.gnorm_log2_ave_gater, gnorm_log2_ave_gater_up)]

        # grad2
        gnorm2_up = [rho * gn2 + (1. - rho) * (g ** 2.) for gn2,g in zip(self.gnorm2, gs)]
        updates = updates + zip(self.gnorm2, gnorm2_up)

        logger.debug('Compiling grad function')
        st = time.time()
        self.train_fn = theano.function(
            [], outs, name='train_function',
            updates = updates,
            givens = zip(model.inputs, loc_data))
        logger.debug('took {}'.format(time.time() - st))

        new_params = [p - (TT.sqrt(dn2 + eps) / TT.sqrt(gn2 + eps)) * g
                for p, g, gn2, dn2 in
                zip(model.params, self.gs, self.gnorm2, self.dnorm2)]

        updates = zip(model.params, new_params)
        # d2
        d2_up = [(dn2, rho * dn2 + (1. - rho) *
            (((TT.sqrt(dn2 + eps) / TT.sqrt(gn2 + eps)) * g) ** 2.))
            for dn2, gn2, g in zip(self.dnorm2, self.gnorm2, self.gs)]
        updates = updates + d2_up

        self.update_fn = theano.function(
            [], [], name='update_function',
            allow_input_downcast=True,
            updates = updates)

        self.old_cost = 1e20
        self.schedules = model.get_schedules()
        self.return_names = self.prop_names + \
                ['cost',
                        'error',
                        'time_step',
                        'whole_time', 'lr', 
                'cutoff']
        self.prev_batch = None

    def __call__(self):
        df_st = time.time()
        batch = self.data.next()
        df_et = time.time()
        assert batch

        # Perturb the data (! and the model)
        if isinstance(batch, dict):
            batch = self.model.perturb(**batch)
        else:
            batch = self.model.perturb(*batch)
        # Load the dataset into GPU
        # Note: not the most efficient approach in general, as it involves
        # each batch is copied individually on gpu
        if isinstance(batch, dict):
            for gdata in self.gdata:
                gdata.set_value(batch[gdata.name], borrow=True)
        else:
            for gdata, data in zip(self.gdata, batch):
                gdata.set_value(data, borrow=True)
        # Run the trianing function
        g_st = time.time()
        lr_val = self.lr.get_value()
        cutoff_val = self.cutoff.get_value()
        rvals = self.train_fn()
        for schedule in self.schedules:
            schedule(self, rvals[-1])
        self.update_fn()
        g_ed = time.time()
        self.state['lr'] = lr_val
        cost = rvals[-1]
        self.old_cost = cost
        whole_time = time.time() - self.step_timer
        if self.step % self.state['trainFreq'] == 0:
            msg = '.. iter %4d cost %.3f'
            vals = [self.step, cost]
            for dx, prop in enumerate(self.prop_names):
                msg += ' '+prop+' %.2e'
                vals += [float(numpy.array(rvals[dx]))]
            msg += ' dload %s step time %s whole time %s lr %.2e co %.2e'
            vals += [print_time(df_et-df_st),
                     print_time(g_ed - g_st),
                     print_time(time.time() - self.step_timer),
                     lr_val, cutoff_val]
            print msg % tuple(vals)
        self.step += 1
        ret = dict([('cost', float(cost)),
                    ('error', float(cost)),
                       ('lr', lr_val),
                       ('cutoff', cutoff_val),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret
