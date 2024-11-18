from huggingface_hub import login
from datasets import Dataset, concatenate_datasets
from dotenv import load_dotenv
import pandas as pd
import glob
from tqdm import tqdm
import os
from collections import Counter
from transformers import AutoTokenizer

card = "alex-miller/ODABert"
tokenizer = AutoTokenizer.from_pretrained(card, model_max_length=512)


def create_text_column(example):
    textual_data_list = [
        example['title_narrative'],
        example['description_narrative'],
        example['transaction_description_narrative']
    ]
    textual_data_list = [str(textual_data) for textual_data in textual_data_list if textual_data is not None]
    example['text'] = " ".join(textual_data_list).strip()
    return example


def main():
    df_list = list()
    csv_filenames = glob.glob('./data/*.csv')
    for csv_filename in tqdm(csv_filenames):
        df_list.append(pd.read_csv(csv_filename))
    all_data = pd.concat(df_list, ignore_index=True)
    relevance_df = pd.read_csv('output/relevance.csv')
    all_data = all_data.merge(relevance_df, how="left", on="reporting_org_ref")
    all_data = all_data.loc[all_data['disability_sig'].isin([0, 1, 2])]
    all_data = all_data.loc[all_data['disability'] == True]
    all_data = all_data[['title_narrative', 'description_narrative', 'transaction_description_narrative', 'languages', 'disability_sig']]
    dataset = Dataset.from_pandas(all_data, preserve_index=False)
    dataset = dataset.map(create_text_column, num_proc=8)
    dataset = dataset.filter(lambda example: example["text"] != "" and example["text"] is not None and len(example["text"]) > 10)
    # De-duplicate
    df = pd.DataFrame(dataset)
    print(df.shape)
    df = df.drop_duplicates(subset=['text'])
    print(df.shape)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # Add removable column to stratify
    count = Counter()
    count.update(dataset['disability_sig'])
    print(count)

    # Remove some of the full negative examples
    negative = dataset.filter(lambda example: example['disability_sig'] == 0)
    positive = dataset.filter(lambda example: example['disability_sig'] != 0)
    negative = negative.shuffle(seed=42).select(range(positive.num_rows * 2))
    dataset = concatenate_datasets([negative, positive])
    count = Counter()
    count.update(dataset['disability_sig'])
    print(count)

    # Stratify and split
    dataset = dataset.add_column('class_labels', dataset['disability_sig'])
    dataset = dataset.class_encode_column('class_labels').train_test_split(
        test_size=0.2,
        stratify_by_column="class_labels",
        shuffle=True,
        seed=42
    )
    dataset = dataset.remove_columns(["class_labels"])

    dataset.save_to_disk("dataset")
    dataset.push_to_hub("alex-miller/iati-disability-structured")


if __name__ == '__main__':
    load_dotenv()
    HF_TOKEN = os.getenv('HF_TOKEN')
    login(token=HF_TOKEN)
    main()