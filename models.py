from pytorch_lightning import LightningModule
from transformers import AutoConfig, AutoModelForSequenceClassification
from torch import Tensor, argmax
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer, Adam
from sklearn.metrics import classification_report


class SequenceClassifier(LightningModule):
    def __init__(
        self, learning_rate: float, model_name_or_path: str, label_names: list[str]
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.label_names = label_names
        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=len(self.label_names)
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config
        )
        self.loss_function = CrossEntropyLoss()

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.model(input_ids)["logits"]

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        output = self(batch["input_ids"])
        loss = self.loss_function(output, batch["labels"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        output = self(batch["input_ids"])
        loss = self.loss_function(output, batch["labels"])
        self.log("val_loss", loss)
        return loss

    def on_test_epoch_start(self) -> None:
        self.test_preds = []
        self.test_labels = []

    def test_step(self, batch: dict, batch_idx: int) -> None:
        output = self(batch["input_ids"])
        self.test_preds += argmax(output, axis=1).tolist()
        self.test_labels += argmax(batch["labels"], axis=1).tolist()

    def on_test_epoch_end(self) -> None:
        results = classification_report(
            self.test_labels, self.test_preds, digits=4, target_names=self.label_names
        )
        self.logger.experiment.add_text("Results", f"```\n{results}\n```")
        print("Results:\n", results)
