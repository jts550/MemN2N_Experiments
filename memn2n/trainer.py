import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import bAbIDataset
from model import MemN2N
import pdb

class Trainer():
    def __init__(self, config):
        self.train_data = bAbIDataset(config.dataset_dir, config.task)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True)

        self.test_data = bAbIDataset(config.dataset_dir, config.task, train=False)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        settings = {
            "use_cuda": config.cuda,
            "num_vocab": self.train_data.num_vocab,
            "embedding_dim": 20,
            "sentence_size": self.train_data.sentence_size,
            "max_hops": config.max_hops
        }

        print("Longest sentence length", self.train_data.sentence_size)
        print("Longest story length", self.train_data.max_story_size)
        print("Average story length", self.train_data.mean_story_size)
        print("Number of vocab", self.train_data.num_vocab)

        self.device = torch.device("cuda" if config.cuda else "cpu")

        self.mem_n2n = MemN2N(settings)
        self.ce_fn = nn.CrossEntropyLoss(size_average=False)
        self.opt = torch.optim.SGD(self.mem_n2n.parameters(), lr=config.lr, weight_decay=1e-5)
        print(self.mem_n2n)
            
        if config.cuda:
            self.ce_fn   = self.ce_fn.cuda()
            self.mem_n2n = self.mem_n2n.cuda()

        self.start_epoch = 0
        self.config = config

    def fit(self):
        config = self.config
        best_acc = 0
        for epoch in range(self.start_epoch, config.max_epochs):
            loss = self._train_single_epoch(epoch)
            lr = self._decay_learning_rate(self.opt, epoch)

            if (epoch+1) % 10 == 0:
                train_acc = self.evaluate("train")
                test_acc = self.evaluate("test")
                print(epoch+1, loss, train_acc, test_acc)
                if test_acc > best_acc:
                    best_acc = test_acc
        print(train_acc, test_acc)
        return best_acc

    def load(self, directory):
        pass

    def evaluate(self, data="test", ensemble = False):
        correct = 0
        pred_prob_all = np.zeros((0, 85))
        answers = np.zeros(0)
        loader = self.train_loader if data == "train" else self.test_loader
        for step, (story, query, answer) in enumerate(loader):
            story = story.to(self.device)
            query = query.to(self.device)
            answer = answer.to(self.device)

            pred_prob = self.mem_n2n(story, query)[1]
            if ensemble:
                pred_prob_all = np.concatenate((pred_prob_all, pred_prob.data), 0)
                answers = np.concatenate((answers, answer.data), 0)
            pred = pred_prob.data.max(1)[1] # max func return (max, argmax)
            correct += pred.eq(answer.data).cpu().sum()
        acc = correct / len(loader.dataset)
        if ensemble:
            return acc, correct, pred_prob_all, answers
        return acc

    def _train_single_epoch(self, epoch):
        config = self.config
        num_steps_per_epoch = len(self.train_loader)
        for step, (story, query, answer) in enumerate(self.train_loader):
            story = story.to(self.device)
            query = query.to(self.device)
            answer = answer.to(self.device)
        
            self.opt.zero_grad()
            loss = self.ce_fn(self.mem_n2n(story, query)[0], answer)
            loss.backward()

            self._gradient_noise_and_clip(self.mem_n2n.parameters(),
                noise_stddev=1e-3, max_clip=config.max_clip)
            self.opt.step()

        return loss.item()

    def _gradient_noise_and_clip(self, parameters,
                                 noise_stddev=1e-3, max_clip=40.0):
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        nn.utils.clip_grad_norm_(parameters, max_clip)

        for p in parameters:
            noise = torch.randn(p.size()) * noise_stddev
            if self.config.cuda:
                noise = noise.cuda()
            p.grad.data.add_(noise)

    def _decay_learning_rate(self, opt, epoch):
        decay_interval = self.config.decay_interval
        decay_ratio    = self.config.decay_ratio

        decay_count = max(0, epoch // decay_interval)
        lr = self.config.lr * (decay_ratio ** decay_count)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        return lr
