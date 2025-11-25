from transformers import TrainerCallback
import os


class KeepBestModelCallback(TrainerCallback):
    def __init__(
        self,
        best_model_dir="./best_model",
        metric_name="eval_loss",
        greater_is_better=False,
    ):
        self.best_model_dir = best_model_dir
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = float("inf") if not greater_is_better else float("-inf")
        self.best_epoch = None

        # Создаем папку для лучшей модели
        os.makedirs(best_model_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            return

        current_metric = metrics[self.metric_name]
        current_epoch = state.epoch

        is_better = (
            current_metric > self.best_metric
            if self.greater_is_better
            else current_metric < self.best_metric
        )

        if is_better:
            self.best_metric = current_metric
            self.best_epoch = current_epoch

            # Сохраняем лучшую модель
            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")

            if model is not None:
                # Сохраняем модель
                model.save_pretrained(
                    self.best_model_dir,
                )

                # Сохраняем токенайзер если есть
                if tokenizer is not None:
                    tokenizer.save_pretrained(self.best_model_dir)

                # Сохраняем информацию о метрике
                with open(
                    os.path.join(self.best_model_dir, "best_metric.txt"), "w"
                ) as f:
                    f.write(f"{self.metric_name}: {current_metric}\n")
                    f.write(f"epoch: {state.epoch}\n")
                    f.write(f"step: {state.global_step}\n")

    def on_train_end(self, args, state, control, **kwargs):
        print("------ Обучение завершено ------")
        print(f"Лучшая эпоха: {self.best_epoch}")
        print(f"Лучший {self.metric_name}: {self.best_metric:.4f}")
