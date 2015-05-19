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
            
        if state['cutoff_adapt']:
            if 'cutoff_to_mean' not in state:
                state['cutoff_to_mean'] = True
            if 'cutoff_stdevs' not in state:
                state['cutoff_stdevs'] = 4.0
            if 'cutoff_rho' not in state:
                state['cutoff_rho'] = 0.99
        
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

        self.burn_in_steps = self.state.get('burn_in_steps',0)
        if self.burn_in_steps < 0:
            self.burn_in_steps = numpy.ceil(1.0/(1.0-self.state['adarho']))
        if state['cutoff_adapt']:
            #start these at zero, since the initial cutoff is 1 and scaled gradient norm is max 1
            self.gnorm_log_ave = theano.shared( numpy.float32(0), name='gnorm_log_ave')
            #nasty init trick to have a low stdev at the beginning
            self.gnorm_log2_ave = theano.shared( numpy.float32(0), name='gnorm_log2_ave' )
            self.gnorm_log2_ave_gater = theano.shared( numpy.float32(0), name='gnorm_log2_ave_gater' )
            self.cutoff_adapt_steps = theano.shared( numpy.float32(0), name='cutoff_adapt_steps' )
            self.cutoff = theano.shared( numpy.float32(state['cutoff']), name='cutoff' )
            self.cutoff_level = theano.shared( numpy.float32(state['cutoff']), name='cutoff_level' )
            logger.debug("Will use adaptive cutoff computation")
        else:
            self.cutoff = theano.shared( numpy.float32(state['cutoff'], name='cutoff') )
            self.cutoff_level = self.cutoff
            

        ###################################
        # Step 1. Compile training function
        ###################################
        logger.debug('Constructing grad function')
        loc_data = self.gdata
                
        scaled_grads = [g*self.lr for g in model.param_grads]
        
        sum_grad_norms = 0.0
        for scaled_grad,p in zip(scaled_grads, model.params): 
            if p in model.exclude_params_for_norm:
                continue
            scaled_grad_norm = TT.sum(scaled_grad**2)
            if self.state.get('monitor_grad_param_magnitudes'):
                scaled_grad_norm.name = p.name + '_grad_norm'
                model.properties.append((scaled_grad_norm.name, scaled_grad_norm))
                param_norm = TT.sum(p**2)
                param_norm.name = p.name + '_param_norm'
                model.properties.append((param_norm.name, param_norm))
            sum_grad_norms += scaled_grad_norm
        
        norm_gs = TT.sqrt(sum_grad_norms)        
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
                    tmpg = TT.switch(TT.ge(norm_gs, c), g*self.cutoff_level/norm_gs, g)
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
            cut_adapt_step_up = self.cutoff_adapt_steps + 1.0
            #quickly fill the running averages
            
            cut_rho_mean = TT.minimum(numpy.float32(self.state['cutoff_rho']), 
                                 self.cutoff_adapt_steps/cut_adapt_step_up)
            if self.burn_in_steps > 0: #if we do the burn in, take the fast converging mean
                cut_rho_mean2= cut_rho_mean
            else: #else start from 0
                cut_rho_mean2= numpy.float32(self.state['cutoff_rho'])
            gnorm_log = TT.log(norm_gs)
            #here we quiclky converge the mean
            gnorm_log_ave_up = (cut_rho_mean*self.gnorm_log_ave + 
                                (numpy.float32(1.)-cut_rho_mean) * gnorm_log)
            #this can wait as it starts from 0 anyways!
            gnorm_log2_ave_up = (cut_rho_mean2*self.gnorm_log2_ave + 
                                 (numpy.float32(1.)-cut_rho_mean2) * gnorm_log**2)         
            
            cutoff_up = TT.exp(gnorm_log_ave_up +
                               #self.gnorm_log2_ave_gater * 
                               TT.sqrt(TT.maximum(numpy.float32(0.0),
                                                  gnorm_log2_ave_up - gnorm_log_ave_up**2
                                              )
                                   ) * numpy.float32(state['cutoff_stdevs'])  
                           )
            if state['cutoff_to_mean']:
                cutoff_level_up = TT.exp(gnorm_log_ave_up)
            else:
                cutoff_level_up = cutoff_up
            #the idea is that only when we have a decent estimate of the stdew the gater will be 1 as they have the same decay constant
            gnorm_log2_ave_gater_up = TT.cast((cut_rho_mean2*self.gnorm_log2_ave_gater + (1.-cut_rho_mean2)*numpy.float32(1.0)), theano.config.floatX)
            
            if 0:
                logger.warn("Not enforcing cutoff max value!")
                cutoff_up = TT.minimum(numpy.float32(state['cutoff_max']),
                                       (cut_rho_mean2*self.cutoff + (1.-cut_rho_mean2) * cutoff_up
                                        ))
            
            updates  = updates + [(self.gnorm_log_ave, TT.switch(notfinite, self.gnorm_log_ave, gnorm_log_ave_up)),
                                  (self.gnorm_log2_ave, TT.switch(notfinite, self.gnorm_log2_ave, gnorm_log2_ave_up)),
                                  (self.cutoff, TT.switch(notfinite, self.cutoff, cutoff_up)),
                                  (self.gnorm_log2_ave_gater, gnorm_log2_ave_gater_up),
                                  (self.cutoff_level, TT.switch(notfinite, self.cutoff_level, cutoff_level_up)),
                                  (self.cutoff_adapt_steps, cut_adapt_step_up)]

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
        
        model_updates = zip(model.params, new_params)
        
        model_updates = model.censor_updates(model_updates)         
        
        # d2
        d2_up = [(dn2, rho * dn2 + (1. - rho) *
            (((TT.sqrt(dn2 + eps) / TT.sqrt(gn2 + eps)) * g) ** 2.))
            for dn2, gn2, g in zip(self.dnorm2, self.gnorm2, self.gs)]

        self.update_fn = theano.function(
            [], [], name='update_function',
            allow_input_downcast=True,
            updates = model_updates + d2_up)
        
        #during burn in we don't update the parameters and don't adapt the cutoff.
        self.burn_in_fn = theano.function(
            [], [], name='burn_in_function',
            allow_input_downcast=True,
            updates = d2_up + [(self.cutoff, self.state['cutoff'])])
        

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
        if self.burn_in_steps >= 0:
            print '.. burn in to go: %d ' %(self.burn_in_steps,),
            self.burn_in_steps -= 1
            self.burn_in_fn()
        else:
            self.update_fn()

        g_ed = time.time()
        self.state['lr'] = lr_val
        self.state['cutoff'] = cutoff_val
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
        if self.state['cutoff_adapt']:
            print 'gn_log_ave ' , self.gnorm_log_ave.get_value(), ' gn_log2_ave ', self.gnorm_log2_ave.get_value(), ' gs ', self.cutoff_adapt_steps.get_value()
        ret = dict([('cost', float(cost)),
                    ('error', float(cost)),
                       ('lr', lr_val),
                       ('cutoff', cutoff_val),
                       ('time_step', float(g_ed - g_st)),
                       ('whole_time', float(whole_time))]+zip(self.prop_names, rvals))
        return ret
