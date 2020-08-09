---
layout: post
title: Imbalanced Data Classification Using FocalLoss - PyTorch
date: 2020-08-08 13:32:20 +0300
description: sample description # Add post description (optional)
img: imbalance.jpg # Add image post (optional)
fig-caption: imbalance in datasets # Add figcaption (optional)
tags: [data, imbalance]
comments: true
---
If you are a data scientist or an ML engineer, you frequently come across imbalanced datasets and when it comes to classifying them, it's a mess! I had encountered this situation recently and tried out multiple approaches to address the problem. This article is about one of such approaches... **The Focal Loss**

## How do you handle Imbalanced data?
There are a set of well established techniques to handle data imbalance. Some of them include:
* Sampling techniques - Under sampling , Over sampling etc..
* Weighted loss function

A recent addition to this technique is <b>Focal Loss</b>, Originally developed by researchers from Facebook to handle extreme class imbalance between foreground and background in object detection tasks.You could read the original paper [here](https://arxiv.org/abs/1708.02002). 

## What's Focal Loss? What's the big idea?
The idea behind focal loss is quiet simple. Based on the difficulty to classify, data points could be devided into two, <b> easy and hard examples</b>. In the cross-entropy loss setting, even though the loss contributed by easy examples aren't huge,they are not very  close to zero either. Since in an imbalanced dataset we would be having a lot of easy-examples, together their loss would sum up to a significant portion of the total loss.

To give you a concrete example, assue a scenario where we have 1000 data points. 950 belonging to class A and 50 belonging to class B.lets assume 900 of these data points are easy exampls and the rest are hard. For the sake of simplicity, lets assume each easy example contributes a loss of 0.1 units and hard examples contributes a loss of 0.9 units. even if the loss contributed by these examples are as small as 0.1, together they sum up to 0.01 * 900 = 90 , which is the same as loss contributed by hard examples. This makes learning difficult for hard-examples.

Researchers at Facebook had addressed this problem by modifying the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples.

 Quoting the authors:
> We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause. We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a sparse set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training

As shown in the below figure, focal loss modifies the standard cross entropy loss by a factor (1-prob)**gamma . This would make sure that the well classified examples are contributing approximately zero units towards the loss. 

![focal loss graph]({{site.baseurl}}/assets/img/focalloss-graph.jpg)

In the implementation of focal loss, they uses another parameter **alpha** for addressing class weights.

![focal loss equation]({{site.baseurl}}/assets/img/focalloss-equation.png)

## PyTorch Implementation of focal loss
Here is an implementation of multi-class focal loss in PyTorch. It inherits from the **_WeightedLoss** base class.

{% highlight ruby %}

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

{% endhighlight %}

## Does it make a difference?
Lets run some experiments to see if focal loss will actually make a difference or not!

