import torch
import kornia
import torch.nn.functional as F
from torch import exp

def q_abf(imgA, imgB, imgF):
    Tg, kg, Dg = 0.9994, -15, 0.5
    Ta, ka, Da = 0.9879, -22, 0.8

    SA = kornia.filters.sobel(imgA)
    gA = torch.norm(SA, dim=1)
    aA = torch.atan2(SA[:, 0], SA[:, 0])  # Assuming single-channel, use the same channel for both x and y

    SB = kornia.filters.sobel(imgB)
    gB = torch.norm(SB, dim=1)
    aB = torch.atan2(SB[:, 0], SB[:, 0])

    SF = kornia.filters.sobel(imgF)
    gF = torch.norm(SF, dim=1)
    aF = torch.atan2(SF[:, 0], SF[:, 0])

    GAF = torch.where(gA > gF, gF / gA, gA / gF)
    AAF = 1 - torch.abs(aA - aF) / (torch.pi / 2)

    QgAF = Tg / (1 + exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + exp(ka * (AAF - Da)))

    QAF = QgAF * QaAF

    GBF = torch.where(gB > gF, gF / gB, gB / gF)
    ABF = 1 - torch.abs(aB - aF) / (torch.pi / 2)

    QgBF = Tg / (1 + exp(kg * (GBF - Dg)))
    QaBF = Ta / (1 + exp(ka * (ABF - Da)))

    QBF = QgBF * QaBF

    deno = torch.sum(gA + gB)
    nume = torch.sum(QAF * gA + QBF * gB)
    qabf_value = nume / deno

    return qabf_value


def q_abf_loss(imgA, imgB, imgF):
  return -q_abf(imgA, imgB, imgF)