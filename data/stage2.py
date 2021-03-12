import csv

with open('train_labels.csv') as tlf:
    with open('val_labels.csv') as vlf:
        with open('train_val_labels_STAGE2.csv', 'w') as olf:
            train_reader = csv.reader(tlf)
            val_reader = csv.reader(vlf)
            out_writer = csv.writer(olf)

            for row in train_reader:
                out_writer.writerow([*row, 'train'])
            
            for row in val_reader:
                out_writer.writerow([*row, 'val'])
