import math
import torch
import torch.nn as nn

class DepressedPredictor(nn.Module):
    def __init__(self,*args, **kwargs):
        super(DepressedPredictor, self).__init__()

        self.squared_angle_weight = pow(1, 2)
        self.time_weight = 10.0
        self.fit_weight = 0.4
        self.dist_tolerance_strict = 3.0
        self.dist_tolerance_loose = 1.5
        self.angle_tolerance = 5 * math.pi / 180
        self.hardness = 0.995
        self.dist_weight = 0.20
        self.angle_weight = 1.25


    def loss_fuc(self, prediction, projection, target, predicted_time, time_to_predict):
        # TODO set weights and tolerances
        dist_loss = self.cal_dist_loss(projection, target)
        angle_loss = torch.mean(self.cal_angle_loss(projection, target))
        fit_loss = self.dist_weight * dist_loss + self.angle_weight * angle_loss
        dist_loss_suppressed, angle_loss_suppressed, time_loss_suppressed = self.cal_loss(prediction, predicted_time,
                                                                                          target, time_to_predict)
        dist_loss2 = torch.quantile(0.95 * torch.max(nn.functional.relu(dist_loss_suppressed - self.dist_tolerance_strict)), 0.995) +\
                     0.05 * torch.mean(nn.functional.relu(dist_loss_suppressed - self.dist_tolerance_strict / 3))
        # angle_loss2 = torch.max(nn.functional.relu(angle_loss_suppressed - self.angle_tolerance))
        angle_loss2 = torch.quantile(nn.functional.relu(torch.minimum((angle_loss_suppressed - math.pi / 6) * 3, dist_loss_suppressed - self.dist_tolerance_loose / 3)), self.hardness) +\
            torch.mean(nn.functional.relu(torch.minimum((angle_loss_suppressed - math.pi / 6) * 3, dist_loss_suppressed - self.dist_tolerance_loose / 3)))

        time_loss = torch.mean(time_loss_suppressed)
        loss = self.fit_weight * fit_loss + dist_loss2 + angle_loss2 + self.time_weight * time_loss
        # print(f"dist_loss :{dist_loss}\t angle_loss :{angle_loss}\t fit_loss:{fit_loss}\t time_loss :{time_loss} \t ")
        # print(f"dist_loss_suppressed:{torch.max(dist_loss_suppressed)} \t angle_loss_suppressed:{torch.max(angle_loss_suppressed)} \t time_loss_suppressed:{torch.max(time_loss_suppressed)}")
        # print(f"angle_loss2:{angle_loss2} \t dist_loss2: {dist_loss2}")
        # print("loss:", loss)
        return loss, (dist_loss_suppressed, angle_loss_suppressed, torch.mean(predicted_time), fit_loss, dist_loss, angle_loss)

    @staticmethod
    def cal_dist_loss(pred, target):
        return torch.sqrt(torch.mean(torch.square(torch.norm(pred - target, dim=-1))))

    @staticmethod
    def cal_angle_loss(pred, target):
        MIN_DISTANCE_TO_COMPARE_ANGLE = 0.1
        raw_cosine = -nn.functional.cosine_similarity(pred, target, dim=-1)
        modified_cosine = torch.where(torch.logical_and(torch.norm(pred, dim=-1) > MIN_DISTANCE_TO_COMPARE_ANGLE,
                                                        torch.norm(target, dim=-1) > MIN_DISTANCE_TO_COMPARE_ANGLE),
                                      raw_cosine, torch.ones_like(raw_cosine))
        cos_sim = torch.clamp(modified_cosine, -0.9999, 0.9999)
        return torch.acos(cos_sim)

    def cal_loss(self, prediction, predicted_time, target, time_to_predict):
        future_length = torch.tensor(time_to_predict.shape[1], dtype=torch.float32).to(self.device)
        weight = torch.abs(predicted_time - time_to_predict)
        weight = nn.functional.relu(1 / future_length - weight) * future_length
        expected = torch.sum(weight[:, :, None] * target, 1)
        return torch.norm(prediction - expected, dim=-1), self.cal_angle_loss(prediction, expected), 1 - predicted_time
