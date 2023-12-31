import csv

def read_csv_into_dict(filename):
    data = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data[row['filename']] = row
    return data

def append_cols(original_table, data, cols):
    pool = []
    with open(original_table, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for col in cols:
                if row['filename'] in data and col in data[row['filename']]:
                    row[col] = data[row['filename']][col]
            pool.append(row)
    with open(original_table, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        column_names = ['path', 'filename', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER'] + cols
        spamwriter.writerow(column_names)
        for row in pool:
            spamwriter.writerow([row[col_name] if col_name in row else '' for col_name in column_names])


if __name__ == "__main__":
    data = read_csv_into_dict('./data/ADNI.csv')
    cols = ["age", "gender", "apoe", "education", "race",
            "trailA", "trailB", "boston", "digitB", "digitBL", "digitF",
            "digitFL", "animal", "gds", "lm_imm", "lm_del", "mmse",
            "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX", "npiq_ELAT", "npiq_APA",
            "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP",
            "faq_BILLS", "faq_TAXES", "faq_SHOPPING", "faq_GAMES", "faq_STOVE",
            "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL",
            "his_NACCFAM", "his_CVHATT", "his_CBSTROKE","his_HYPERTEN","his_DEPOTHR", "his_PSYCDIS", "his_ALCOHOL","his_SMOKYRS", "his_PACKSPER",
            "benefit"]
    for i in range(5):
        for stage in ['train', 'valid', 'test']:
            append_cols('./CrossValid/cross{}/{}.csv'.format(i, stage), data, cols)





