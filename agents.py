# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 19:42:15 2022

@author: bishopns
"""

import random
import numpy as np


class agent():
    #%% initialize 
    def __init__(self,a_id=None,ntraits=10,nfriends=10):
        self.person_type = random.randint(0,1) # 0 = introvert; 1 = extrovert
        self.person_traits = [2*random.random() for xx in range(ntraits)]#+[self.person_type] # for diff comparison
        self.battery_lvl = random.random() # randomized battery levels
        self.inv_max_prob = 0.1 # max prob to send an invite
        self.battery_influence = 0.1 #influence meeting a person has on battery level
        # ^^ i.e. how much is added or subtracted to battery at each time step ^^
        self.battery_decay = 0.05 # rate at which battery level changes when alone
        # ^^ different b/w person types ^^
        self.num_friends = nfriends # start with zero friends
        self.friends = []
        
        # logistics variables
        self.rec_invites = set()
        self.sent_accepts = set()
        self.rec_accepted_invites = set()
        self.rec_declined_invites = set()
        
        # emotion vars
        self.decline_reduce =  0.05
        self.accept_increase = 0.05
        self.happiness = 0
        
        # fondness dictionary to all other agents
        # will be populated once population is created
        self.fondness = {}
        
        # memory
        self.former_friends = []
        self.fondness_memory = {}
        
        # set individual id of agent
        if a_id is None:
            a_id = random.randint(0,1e100)
        self.agent_id = a_id
        self.log = []
    
    #%% get attributes
    def get_personality(self):
        # return string description of person type #
        personalities = ['introvert','extrovert']
        personality_str = personalities[self.person_type]
        return personality_str
    def get_fondness(self):
        # return fondness in human readable format #
        fond = {k.agent_id:v for k,v in self.fondness.items()}
        return fond
    
    #%% agent functions
    def send_invs(self):
        # logistic function to send invites to friends #
        self.log.append(['send_invs','-'*10])
        for agent,fond in self.fondness.items():
            if fond > 0:
                agent.rec_invites.add(self)
                assert (self in agent.rec_invites)
                self.log.append(['send_invs',agent.agent_id])
    
    def accept_invs(self):
        # logistic function to facilitate the "accept invite" interaction #
        # accept only if friends
        self.log.append(['accept_invs','-'*10])
        for inviter in self.rec_invites:
            if inviter in self.friends:
                inviter.rec_accepted_invites.add(self)
                self.sent_accepts.add(inviter)
                self.log.append(['accept_invs',inviter.agent_id])
            else:
                self.decline_single_inv(inviter)
    
    def decline_single_inv(self,inviter):
        # logistic function to allow declining invites from agents #
        # not in the current agent's friend list #
        self.log.append(['decline_single_inv','-'*10])
        inviter.rec_declined_invites.add(self)
        self.log.append(['decline_single_inv',inviter.agent_id])
    
    def decline_invs(self):
        # logistic funciton to facilitate the "decline invite" interaction #
        self.log.append(['decline_invs','-'*10])
        for inviter in self.rec_invites:
            inviter.rec_declined_invites.add(self)
            self.log.append(['decline_invs',inviter.agent_id])
    
    def reset_invs(self):
        # housekeeping function to reset interaction variables after #
        # the interaction phase of the simulation #
        self.rec_invites = set()
        self.sent_accepts = set()
        self.rec_accepted_invites = set()
        self.rec_declined_invites = set()
    
    #%% emotion management setup    
    def get_init_fondness(self,other):
        # sinlge line of code put into function for readability of other functions #
        fondness = self.fondness_lookup[(self,other)]
        return fondness
    
    def set_init_fondness(self,pop):
        # setup function used to create look up table of all agent initial fondness #
        # to increase optimization for run time and make interactions easier #
        pop_others = [xx for xx in pop if xx!=self]
        pop_others_conn = [xx for xx in pop_others if len(xx.fondness)<xx.num_friends]
        self.log.append(['set_init_fondness','pop others conn',[xx.agent_id for xx in pop_others_conn]])
        
        if len(pop_others_conn) >= self.num_friends:
            self.log.append(['set_init_fondness','reg sel'])
            sel = np.random.choice(pop_others_conn,size=self.num_friends-len(self.fondness),replace=False).tolist()
        else:
            selsize = min(len(pop_others_conn),self.num_friends-len(self.fondness))
            sel = np.random.choice(pop_others_conn,size=selsize,replace=False).tolist()
            self.log.append(['set_init_fondness','special case sel',selsize])
        
        self.log.append(['set_init_fondness','len fond',len(self.fondness),
                         'num friends',self.num_friends,
                         'nfriend-len fond',self.num_friends-len(self.fondness)])
        self.log.append(['set_init_fondness','sel',[xx.agent_id for xx in sel]])
        for other in sel:
            self.fondness[other] = self.get_init_fondness(other)
            self.log.append(['set_init_fondness','reached out to',other.agent_id])
        self.friends = [xx for xx in self.fondness.keys()]
        for friend in self.friends:
            friend.fondness[self] = self.fondness[friend]
            friend.log.append(['set_init_fondness','rec new friend from',self.agent_id])
        self.log.append(['set_init_fondness',[xx.agent_id for xx in self.friends],
                         {k.agent_id:v for k,v in self.fondness.items()}])
        
    #%% emotion management
    def decline_reduce_fondness(self):
        # logistic function to decrease fondness towards agents that decline invites #
        self.log.append(['decline_reduce_fondness','-'*10])
        for dec_inv in self.rec_declined_invites:
            old_val = self.fondness[dec_inv]
            self.fondness[dec_inv] -= self.decline_reduce
            self.log.append(['decline_reduce_fondness',
                             'agent dec_inv {}'.format(dec_inv.agent_id),
                             'old {}, new {}'.format(old_val,self.fondness[dec_inv])])
            
    def decline_reduce_fondness_ingroup(self):
        # logistic function to decrease fondness towards agents that decline invites #
        # taking into account the in-group status of the other agent. used for the #
        # application of in-group event empathy #
        self.log.append(['decline_reduce_fondness','-'*10])
        for dec_inv in self.rec_declined_invites:
            old_val = self.fondness[dec_inv]
            if dec_inv.person_type == self.person_type:
                self.fondness[dec_inv] -= self.decline_reduce
            else:
                self.fondness[dec_inv] -= 0.05
            self.log.append(['decline_reduce_fondness',
                             'agent dec_inv {}'.format(dec_inv.agent_id),
                             'old {}, new {}'.format(old_val,self.fondness[dec_inv])])
            
    def accept_increase_fondness(self):
        # logistic function to increase fondness towards agents that accept invites #
        self.log.append(['accept_increase_fondness','-'*10])
        for acc_inv in self.rec_accepted_invites:
            old_val = self.fondness[acc_inv]
            self.fondness[acc_inv] += self.accept_increase
            self.fondness[acc_inv] = min(1,self.get_init_fondness(acc_inv))
            self.log.append(['accept_increase_fondness',
                             'agent acc_inv {}'.format(acc_inv.agent_id),
                             'old {}, new {}'.format(old_val,self.fondness[acc_inv])])
            
    def unfriend_friend(self,friend):
        # logistic function to remove an agent from the friend list #
        # also involves moving the other agent from "fondness" to "fondness memory" #
        self.log.append(['block_friend',friend.agent_id,'-'*10])
        self.fondness_memory[friend] = self.fondness.pop(friend)
        self.log.append(['block_friend',
                        'curr fondness',self.get_fondness(),
                        'curr fond mem',{k.agent_id:v for k,v in self.fondness_memory.items()}])
        # self.friends.pop(self.friends.index(friend))
        self.friends = [xx for xx in self.fondness.keys()]
        self.former_friends = [xx for xx in self.fondness_memory.keys()]
        self.log.append(['block_friend',
                        'curr friends',[xx.agent_id for xx in self.friends],
                        'curr former friends',[xx.agent_id for xx in self.former_friends]])
        
    def fill_friend_spot(self,pop):
        # logistic function to replace a void space in the friend list for current agent #
        # sub-function for update_friends #
        # select another friend with fondness > 0
        self.log.append(['fill_friend_spot','-'*10])
        pop_others = [xx for xx in pop if xx not in [self]+self.friends]
        pop_others_conn = [xx for xx in pop_others if len(xx.fondness)<xx.num_friends]
        if len(pop_others_conn) > 0:
            sel = np.random.choice(pop_others,size=1,replace=False)[0]
            # for sel in rand_sel:
            self.log.append(['fill_friend_spot','sel',sel.agent_id])
            if sel in self.fondness_memory.keys():
                fond = self.fondness_memory[sel]
            else:
                fond = self.get_init_fondness(sel)
            self.log.append(['fill_friend_spot','fond',fond])
            if fond > 0.0:
                consent,other_fond = self.ask_for_consent(sel)
                if consent == True:
                    self.fondness[sel] = fond
                    self.friends = [xx for xx in self.fondness.keys()]
                    self.log.append(['fill_friend_spot','new friend',sel.agent_id])
                    sel.fondness[self] = other_fond
                    sel.friends = [xx for xx in sel.fondness.keys()]
                
            self.log.append(['fill_friend_spot','new friend',sel.agent_id,'fond',fond])
        
    def replace_friend(self,old_friend,new_friend,fond):
        # logistic function to handle attribute changes that come with a new friend #
        # sub-function for try_new_things #
        if old_friend is not None:
            self.log.append(['replace_friend','old friend',old_friend.agent_id,
                             'new friend',new_friend.agent_id,'-'*10])
            self.unfriend_friend(old_friend)
        else:
            self.log.append(['replace_friend','old friend',old_friend,
                             'new friend',new_friend.agent_id,'-'*10])
        self.fondness[new_friend] = fond
        self.friends = [xx for xx in self.fondness.keys()]
        self.log.append(['replace_friend ^^','friends',[xx.agent_id for xx in self.friends],
                         'former friends',[xx.agent_id for xx in self.former_friends]])
                        
    def update_friends(self,pop,apply_ingroup=False):
        # housekeeping function to ensure that friendships are broken #
        # according to fondness rules and event empathy is properly applied #
        self.log.append(['update_friends','-'*10])
        if apply_ingroup:
            self.decline_reduce_fondness_ingroup()
        else:
            self.decline_reduce_fondness()
        self.accept_increase_fondness()
        for friend in self.friends:
            if self.fondness[friend] < 0.0:
                self.unfriend_friend(friend)
                self.fill_friend_spot(pop)
        if len(self.fondness)<self.num_friends:
            self.fill_friend_spot(pop)
    
    def give_consent_try_new(self,friend_inv):
        # allow another agent to initiate testing friendship current agent #
        # agents will never not give consent, asking and giving consent was #
        # the easiest way to think of the logic to allow the try_new_things #
        # interaction to work properly #
        # part 2 of handshake #
        if friend_inv in self.fondness_memory.keys():
            fond = self.fondness_memory[friend_inv]
        else:
            fond = self.get_init_fondness(friend_inv)
        if len(self.fondness)<1:
            curr_min_fond = 0
        else:
            curr_min_fond = min([xx for k,xx in self.fondness.items()])
        k = None
        if len(self.fondness)>0:
            for k,v in self.fondness.items():
                if v == curr_min_fond:
                    break
        if fond > curr_min_fond:
            consent = True
        else:
            consent = False
        return consent,fond,k
    
    def ask_for_consent_try_new(self,new_friend):
        # ask other agent if current agent can test new frienship #
        # part 1 handshake #
        # k is new friend's friend which they are dropping
        consent,fond,k = new_friend.give_consent_try_new(self)
        return consent,fond,k
    
    def give_consent(self,friend_inv):
        # part 2 of handshake #
        if friend_inv in self.fondness_memory.keys():
            fond = self.fondness_memory[friend_inv]
        else:
            fond = self.get_init_fondness(friend_inv)
        if fond > 0.0 and len(self.fondness)<self.num_friends:
            consent = True
        else:
            consent = False
        return consent,fond
    
    def ask_for_consent(self,new_friend):
        # sub-function for fill_friend_spot #
        # part 1 of handshake #
        consent,fond = new_friend.give_consent(self)
        return consent,fond
        
        
    #%% empathy and happiness
    def empathize_ingroup(self,other,fond_dict):
        # allows for time emapthy only for in-group agents #
        # sub-function for apply_time_empathy #
        init_fond = self.get_init_fondness(other)
        curr_fond = fond_dict[other]
        emp_fondness = curr_fond + (init_fond-curr_fond)*0.001
        fond_dict[other] = emp_fondness
    
    def apply_time_empathy(self,mode=None):
        # time based empathy #
        self.log.append(['empathize','mode',mode,'-'*10])
        if mode == 'in-type':
            for friend in [xx for xx in self.friends if xx.person_type==self.person_type]:
                self.empathize_ingroup(friend,self.fondness)
            for exfriend in [xx for xx in self.former_friends if xx.person_type==self.person_type]:
                self.empathize_ingroup(exfriend,self.fondness_memory)
        elif mode == 'all':
            for friend in self.friends:
                self.empathize_ingroup(friend,self.fondness)
            for exfriend in self.former_friends:
                self.empathize_ingroup(exfriend,self.fondness_memory)
        elif mode == 'None':
            pass
    
    def try_new_things(self,pop,prob=0.05):
        # try new friendship with another agent #
        # if fondness is greater than min fondness towards a friend #
        # for each agent, create new friendship and drop min fondness friends #
        
        # probabilty of testing init_fondness of with a new person
        self.log.append(['try_new_things','prob',prob,'-'*10])
        other_pop = [xx for xx in pop if xx not in [self]+self.friends]
        sel = random.choice(other_pop)
        if sel in self.fondness_memory.keys():
            fond = self.fondness_memory[sel]
        else:
            fond = self.get_init_fondness(sel)
        if len(self.fondness)<1:
            curr_min_fond = 0
        else:
            curr_min_fond = min([xx for k,xx in self.fondness.items()])
        if fond > curr_min_fond:
            # find key for min fond value
            k = None
            for k,v in self.fondness.items():
                if v == curr_min_fond:
                    break
            consent,other_fond,other_old_friend = self.ask_for_consent_try_new(sel)
            if consent == True:
                self.replace_friend(k,sel,fond)
                sel.replace_friend(other_old_friend,self,other_fond)
                if k is not None:
                    self.log.append(['try_new_things','replace','old',k.agent_id,'new',sel.agent_id])
                else:
                    self.log.append(['try_new_things','replace','old',k,'new',sel.agent_id])
            
    def update_happiness(self):
        # housekeeping function for adjusting happiness after friendships are changed #
        self.log.append(['get_happiness','-'*10])
        happiness = 0
        friends = [xx for xx in self.fondness.keys()]
        for friend in friends:
            happiness += self.get_init_fondness(friend)
        self.happiness = happiness
        self.log.append(['get_happiness',happiness])
    
    
    #%% battery functions
    def battery_action_introvert(self,action='None'):
        # function for facilitating interactions if current agent is introvert #
        self.log.append(['battery_action_introvert',action])
        if action not in ('None','receive','send'):
            print('action unknown in agent {}:{}; performing None action'.format(self.agent_id,self))
        
        prob_accept_inv = self.battery_lvl # prob to accept proportional to battery lvl
        prob_send_inv = self.inv_max_prob * self.battery_lvl # prob to send prop to battery level * k
        if action=='receive':
            if random.random() < prob_accept_inv:
                # self.set_battery_level()
                self.accept_invs()
            else:
                self.decline_invs()
        if action=='send':
            if random.random() < prob_send_inv:
                self.send_invs()
    
    def battery_action_extrovert(self,action='None'):
        # function for facilitating interactions if current agent is extrovert #
        self.log.append(['battery_action_extrovert',action])
        if action not in ('None','receive','send'):
            print('action unknown in agent {}:{}; performing None action'.format(self.agent_id,self))
        
        prob_accept_inv = 1-(self.battery_lvl*0.0) # prob to accept inverse prop to battery lvl * 0.5
        prob_send_inv = self.inv_max_prob * (1-self.battery_lvl) # prob to send inv prop to battery level * k
        if action=='receive':
            if random.random() < prob_accept_inv:
                self.accept_invs()
            else:
                self.decline_invs()
        if action=='send':
            if random.random() < prob_send_inv:
                self.send_invs()
            
    def battery_action(self,action):
        # core function for facilitating all agent interactions #
        self.log.append(['battery_action','-'*10])
        pt = self.person_type
        if pt == 0:
            self.battery_action_introvert(action)
        if pt == 1:
            self.battery_action_extrovert(action)
    
    def set_battery_level(self):
        # handles logistics of changing battery levels #
        # sub-function for update_batter_level #
        self.log.append(['set_battery_level','start ',self.battery_lvl])
        bi = self.battery_influence # default is 0.1
        for inviter in self.sent_accepts:
            pt_val = -1*(-1)**(self.person_type)
            # test
            if self.person_type==0:
                assert(pt_val==-1)
            elif self.person_type==1:
                assert(pt_val==1)
            # test            
            # inc or dec by 0.1 out of cap of 1
            self.battery_lvl += bi*pt_val
            self.log.append(['set_battery_level','sent_accepts ',
                             'inviter {}'.format(inviter.agent_id),
                             'battery {}'.format(self.battery_lvl)])
        for invites in self.rec_accepted_invites:
            pt_val = -1*(-1)**(self.person_type)
            # inc or dec by 0.1 out of cap of 1
            self.battery_lvl += bi*pt_val
            self.log.append(['set_battery_level','rec_accepts ',
                             'invites {}'.format(invites.agent_id),
                             'battery {}'.format(self.battery_lvl)])
        
    def update_battery_level(self):
        # logistic funciton for updating the battery level of current agent #
        # based on interactions in last interaction phase of sim #
        self.log.append(['update_battery_level','start ',self.battery_lvl])
        self.set_battery_level()
        bl = self.battery_lvl
        if bl == -1000:
            self.battery_lvl *= 1
            self.log.append(['update_battery_level','nothing ',self.battery_lvl])
        elif bl < 0 or bl > 1:
            self.battery_lvl = max(0.0,min(self.battery_lvl,1.0))
            self.log.append(['update_battery_level','clip ',self.battery_lvl])
        else:
            pt_val = -1*(-1)**(self.person_type) # -1 for introvert, +1 for extrovert
            self.battery_lvl += (-1*pt_val) * self.battery_decay * (1-self.battery_lvl)
            self.battery_lvl = max(0.0,min(self.battery_lvl,1.0))
            self.log.append(['update_battery_level',
                             'change',(-1*pt_val) * self.battery_decay,
                             'update ',self.battery_lvl])