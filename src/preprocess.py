import json
import os

from tap import Tap

from parse.generate_train_examples import generate_train_examples


class Config(Tap):
    # data
    data_dir: str = "data"
    save_dir: str = "preprocessed_data"
    corpus: str = "rstdt"  # rstdt or instrdt
    train_file_name: str = "train.json"
    valid_file_name: str = "valid.json"
    test_file_name: str = "test.json"


def main(config: Config):
    file_names = {
        "train": config.train_file_name,
        "valid": config.valid_file_name,
        "test": config.test_file_name,
    }
    for split, file_name in file_names.items():
        # preprocess
        raw_data = json.load(
            open(os.path.join(config.data_dir, config.corpus, file_name))
        )
        preprocessed_datasets = generate_train_examples(raw_data, corpus=config.corpus)
        # save
        os.makedirs(os.path.join(config.save_dir, config.corpus), exist_ok=True)
        for key in preprocessed_datasets.keys():
            save_path = os.path.join(
                config.save_dir, config.corpus, split, f"{key}.json"
            )
            if os.path.exists(save_path):
                print(f"Skip: {save_path} already exists.")
                continue

            preprocessed_datasets[key].to_json(save_path)


if __name__ == "__main__":
    config = Config().parse_args()
    main(config)
