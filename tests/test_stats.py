from qtaim_embed.models.link_pred.losses import (
    HingeMetric, 
    CrossEntropyMetric, 
    AUCMetric, 
    MarginMetric, 
    F1Metric, 
    AccuracyMetric, 
    compute_loss_hinge, 
    compute_loss_margin, 
    compute_loss_cross_entropy, 
    compute_auc, 
    compute_accuracy
)
 
import torch


class TestLinkPredMetrics:

    labels = torch.tensor([0, 0, 0, 1, 1])
    preds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    labels2 = torch.tensor([0, 1, 1, 1, 1])
    preds2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    
    def test_hinge(self): 
        hinge = HingeMetric()
        hinge.reset()
        hinge.update(self.preds, self.labels)
        hinge_metric = hinge.compute()


        assert hinge_metric == compute_loss_hinge(self.preds, self.labels).mean()

        hinge.update(self.preds2, self.labels2)
        hinge.update(self.preds2, self.labels2)
        
        concat_hinge = torch.concat([
            compute_loss_hinge(self.preds, self.labels), 
            compute_loss_hinge(self.preds2, self.labels2), 
            compute_loss_hinge(self.preds2, self.labels2)
            ]
        )
        assert torch.allclose(concat_hinge.mean(), hinge.compute())

        hinge.reset()

    def test_accuracy(self):
        
        accuracy = AccuracyMetric()
        accuracy.reset()
        accuracy.update(self.preds, self.labels)
        accuracy_metric = accuracy.compute()

        assert accuracy_metric == compute_accuracy(self.preds, self.labels)

        accuracy.update(self.preds2, self.labels2)
        accuracy.update(self.preds2, self.labels2)

        concat_preds = torch.concat([
                self.preds, 
                self.preds2,
                self.preds2
            ]
        )
        concat_labels = torch.concat([
                self.labels,
                self.labels2,
                self.labels2
            ]
        )
        concat_accuracy = compute_accuracy(concat_preds, concat_labels)

        assert torch.allclose(concat_accuracy, accuracy.compute())

        accuracy.reset()
        
    def test_auc(self):

        auc = AUCMetric()
        auc.reset()
        auc.update(self.preds, self.labels)
        auc_metric = auc.compute()

        assert auc_metric == compute_auc(self.preds, self.labels)

        auc.update(self.preds2, self.labels2)
        auc.update(self.preds2, self.labels2)

        concat_preds = torch.concat([
                self.preds, 
                self.preds2,
                self.preds2
            ]
        )
        concat_labels = torch.concat([
                self.labels,
                self.labels2,
                self.labels2
            ]
        )
        
        concat_auc = compute_auc(concat_preds, concat_labels)

        #auc.reset()

    def test_margin(self):
        margin = MarginMetric()
        margin.reset()
        margin.update(self.preds, self.labels)
        margin_metric = margin.compute()

        assert margin_metric == compute_loss_margin(self.preds, self.labels).mean()

        margin.update(self.preds2, self.labels2)
        margin.update(self.preds2, self.labels2)

        concat_margin = torch.concat([
            compute_loss_margin(self.preds, self.labels), 
            compute_loss_margin(self.preds2, self.labels2), 
            compute_loss_margin(self.preds2, self.labels2)
            ]
        )
        assert torch.allclose(concat_margin.mean(), margin.compute())

        margin.reset()


    def test_cross_entropy(self):
        cross_entropy = CrossEntropyMetric()
        cross_entropy.reset()
        cross_entropy.update(self.preds, self.labels)
        cross_entropy_metric = cross_entropy.compute()

        assert cross_entropy_metric == compute_loss_cross_entropy(self.preds, self.labels).mean()

        cross_entropy.update(self.preds2, self.labels2)
        cross_entropy.update(self.preds2, self.labels2)

        concat_cross_entropy = torch.concat([
            compute_loss_cross_entropy(self.preds, self.labels), 
            compute_loss_cross_entropy(self.preds2, self.labels2), 
            compute_loss_cross_entropy(self.preds2, self.labels2)
            ]
        )
        assert torch.allclose(concat_cross_entropy.mean(), cross_entropy.compute())

        cross_entropy.reset()





