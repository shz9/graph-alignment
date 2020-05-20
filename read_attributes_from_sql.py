import os
import subprocess
import pandas as pd


table_names = []

output = subprocess.check_output('grep "CREATE TABLE" ./data/multions_2015-08-27.sql', shell=True)

for tbl in output.decode().split('\n'):
    try:
        table_names.append(tbl.split('`')[1])
    except Exception:
        pass

output = subprocess.check_output('grep -n "INSERT INTO" ./data/multions_2015-08-27.sql', shell=True)

insert_ln = []

for ln in output.decode().split('\n'):

    try:
        insert_ln.append((int(ln.split(':')[0]), ln.split("`")[1]))
    except Exception:
        pass

insert_ln = dict(insert_ln)

data_tables = {tbl: {} for tbl in table_names}

read = False

with open("./data/multions_2015-08-27.sql", "r") as f:
    for i, line in enumerate(f, 1):
        if i in insert_ln:
            if 'rows' not in data_tables[insert_ln[i]]:
                data_tables[insert_ln[i]]['columns'] = line.split('(')[1].strip()[:-1].replace('`', '').replace(' ', '').split(',')
                data_tables[insert_ln[i]]['rows'] = []
            current_table = insert_ln[i]
            read = True
        elif line.strip() == 'VALUES':
            continue
        elif read:
            if line.strip()[-1] == ';':
                read = False
                line = line[:-1]

            fin_line = [s.strip().replace("'", "") for s in line.strip()[1:-2].split("',")]
            data_tables[current_table]['rows'].append(dict(zip(data_tables[current_table]['columns'],
                                                               fin_line)))

for k, v in data_tables.items():
    df = pd.DataFrame(v['rows'])
    df.to_csv(os.path.join("./data/graph_attributes/", k + ".csv"))
