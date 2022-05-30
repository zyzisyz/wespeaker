#!/usr/bin/env python3
# coding=utf-8
# Author: Hongji Wang

import wespeaker.models.tdnn as tdnn
import wespeaker.models.ecapa_tdnn as ecapa_tdnn
import wespeaker.models.resnet as resnet
import wespeaker.models.mfa_conformer as mfa_conformer

def get_speaker_model(model_name: str):
    if model_name.startswith("XVEC"):
        return getattr(tdnn, model_name)
    elif model_name.startswith("ECAPA_TDNN"):
        return getattr(ecapa_tdnn, model_name)
    elif model_name.startswith("ResNet"):
        return getattr(resnet, model_name)
    elif model_name.startswith("MFA_Conformer"):
        return getattr(mfa_conformer, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
