import csv
import pandas as pd

# tables = ['NACC', 'ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO', 'NIFD', 'PPMI', 'AIBL', 'OASIS', 'FHS', 'Stanford']
# tables = ['ADNI1', 'ADNI2', 'ADNI3', 'ADNIGO']
tables = ['ADNI']
column_names = ['path', 'filename', 'NC', 'MCI', 'DE', 'COG', 'AD', 'PD', 'FTD', 'VD', 'DLB', 'PDD', 'ADD', 'ALL', 'OTHER']

def read_csv_dict(content, csv_table):
    with open(csv_table, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            content.append(row)

content = []

for table in tables:
    csv_path = table + '.csv'
    read_csv_dict(content, csv_path)

with open('all.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(column_names)
    for row in content:
        row['path'] = '/home/huangyunyou/amount_data/new_data/'
        if row['NC'] == '1':
            row['ALL'] = '0'
        elif row['MCI'] == '1':
            row['ALL'] = '1'
        elif row['AD'] == '1':
            row['ALL'] = '2'
        elif row['ADD'] == '0':
            row['ALL'] = '3'
        spamwriter.writerow([row[col_name] if col_name in row else '' for col_name in column_names])

#mri_only有部分重复
all = pd.read_csv('all.csv')

all_datatype = []
for index in all.dtypes.tolist():
    print(index)
    all_datatype.append(index)
print(all.dtypes)
all_datatype = pd.DataFrame(all_datatype)
all_datatype.to_csv("all_datatype.csv", index=0)

all.drop_duplicates(subset=['filename'],keep='first',inplace=True)
all.to_csv('mri_all.csv', index=0)