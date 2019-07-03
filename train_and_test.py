# from gym.envs.toy_text.taxi import TaxiEnv
import gym
import time
import os
import numpy as np
import random


class Taxi(object):
    def __init__(self):
        self.env = gym.make('Taxi-v2')
        self.env.reset()

        self.observation=-1
        self.reward=-1
        self.done=-1
        self.info=-1

        self.env

    def render(self, control=-1, display=0):
        if control == -1:
            control = self.env.action_space.sample()
        if display != 0:
            os.system('cls')
            self.env.render()
            time.sleep(0.1)
        self.observation, self.reward, self.done, self.info = self.env.step(
            control)

    def end(self):
        self.env.close()

    def train(self, alpha, gamma, epsilon, step,display=0):
        '''train and save data
        Args:
            alpha: Learning rate
            gamma: Attenuation value
            epsilon: Epsilon greedy
            step: Train step 
        '''
        L = 50
        N = step
        data = np.zeros((500, 6), np.float)

        first_step = random.randrange(0, 6)
        self.render(first_step)
        for i in range(step):
            use_step=0
            while True:
                column = data[self.observation]
                last_observation = self.observation
                #Calculate Eesilon greedy
                greedy=1
                if random.random()>epsilon:
                    greedy=1

                    if np.min(column) == np.max(column):
                        this_step = random.randrange(0, 6)
                    else:
                        this_step = np.random.choice(
                            np.where(column == np.max(column))[0])
                else:
                    this_step=random.randrange(0,6)
                self.render(this_step,display)
                if self.done == True:
                    self.env.reset()
                    break
                else:
                    use_step+=1
                
                if greedy == 1:
                    data[last_observation][this_step] = (
                        1-alpha)*data[last_observation][this_step]+alpha*(self.reward+gamma*np.max(data[self.observation]))

            print("{{{0}>{1}}} {2}%  use step:{3}".format('='*round(i*L/N),
                                            '.'*round((N-i)*L/N), round(i*100/N),use_step), end="\r")
        print('train complete')
        np.save('result.npy', data)
        self.result = data

    def run(self, step):
        # self.env=gym.make('Taxi-v2')
        self.data = np.load('result.npy')
        self.env.reset()
        first_step = random.randrange(0, 6)
        self.render(first_step)
        for s in range(step):
            column = self.data[self.observation]
            if np.min(column) == np.max(column):
                this_step = random.randrange(0, 6)
            else:
                this_step = np.random.choice(
                    np.where(column == np.max(column))[0])
            self.render(this_step, display=1)
            if self.done == True:
                print('complete, step:{0}'.format(s))
                return
        print('fail')

    def test(self):
        pass


if __name__ == "__main__":
    taxi = Taxi()
    taxi.train(0.8,0.7,0.15,10000)
    taxi.run(100)
