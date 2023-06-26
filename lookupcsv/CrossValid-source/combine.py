import csv

# tables = ['NACC', 'ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO', 'NIFD', 'PPMI', 'AIBL', 'OASIS', 'FHS', 'Stanford']
tables = ['ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO']
# column_names = ['path', 'filename', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER']

column_names = ['path', 'filename', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER',"age", "gender", "education",
                "trailA", "trailB", "boston", "digitB", "digitBL", "digitF",
                "digitFL", "animal", "gds", "lm_imm", "lm_del", "mmse",
                "npiq_DEL", "npiq_HALL", "npiq_AGIT", "npiq_DEPD", "npiq_ANX", "npiq_ELAT", "npiq_APA",
                "npiq_DISN", "npiq_IRR", "npiq_MOT", "npiq_NITE", "npiq_APP",
                "faq_BILLS", "faq_TAXES", "faq_SHOPPING", "faq_GAMES", "faq_STOVE",
                "faq_MEALPREP", "faq_EVENTS", "faq_PAYATTN", "faq_REMDATES", "faq_TRAVEL",
                "his_NACCFAM", "his_CVHATT", "his_CVAFIB", "his_CVANGIO", "his_CVBYPASS", "his_CVPACE",
                "his_CVCHF", "his_CVOTHR", "his_CBSTROKE", "his_CBTIA", "his_SEIZURES", "his_TBI",
                "his_HYPERTEN", "his_HYPERCHO", "his_DIABETES", "his_B12DEF", "his_THYROID", "his_INCONTU", "his_INCONTF",
                "his_DEP2YRS", "his_DEPOTHR", "his_PSYCDIS", "his_ALCOHOL",
                "his_TOBAC100", "his_SMOKYRS", "his_PACKSPER", "his_ABUSOTHR",
                "COG_score", "ADD_score"]

def read_csv_dict(content, csv_table):
    with open(csv_table, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content.append(row)

content = []

for table in tables:
    if table == 'NACC':
        csv_path = '../dataset_table/NACC_ALL/' + table + '.csv'
    else:
        csv_path = '../dataset_table/' + table + '/' + table + '.csv'
    read_csv_dict(content, csv_path)

with open('all.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(column_names)
    for row in content:
        if row['NC'] == '1':
            row['ALL'] = '0'
        elif row['MCI'] == '1':
            row['ALL'] = '1'
        elif row['AD'] == '1':
            row['ALL'] = '2'
        elif row['ADD'] == '0':
            row['ALL'] = '3'
        spamwriter.writerow([row[col_name] if col_name in row else '' for col_name in column_names])