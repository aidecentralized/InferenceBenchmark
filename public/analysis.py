import config_utils as utils
import os
import csv
import json


def name_to_index(given_list):
    d = dict()
    for i in range(len(given_list)):
        d[given_list[i]] = i
    return d

class Database(object):

    def __init__(self, Train=None):
        self.config = utils.load_config()
        self.prediction_attribute = self.config["prediction_attribute"]
        self.protected_attribute = self.config["protected_attribute"]
        self.fields = None
        self.csv_fields = None
        self.mappings = None
        self.rows = []
        self.label_mapping = {}
        self.label_mapping["race"] = {"East Asian": 0,
                                      "Indian": 1,
                                      "Black": 2,
                                      "White": 3,
                                      "Middle Eastern": 4,
                                      "Latino_Hispanic": 5,
                                      "Southeast Asian": 6}
        self.label_mapping["gender"] = {"Male": 0, "Female": 1}
        self.label_mapping["r_gender"] = self.reverse_mapping(self.label_mapping['gender'])
        self.label_mapping["r_race"] = self.reverse_mapping(self.label_mapping['race'])
        self.label_csv = None
        self.split_index = 0
        self.config["path"] = self.config["dataset_path"]
        if Train:
            label_csv = self.config["path"] + "fairface_label_train.csv"
            self.config["path"] += "train"
            self.split_index = len("train/")
        else:
            label_csv = self.config["path"] + "fairface_label_val.csv"
            self.config["path"] += "val"
            self.split_index = len("val/")
        self.label_csv = label_csv
        self.json_dict = None
        self.import_from_csv()
            

    def reverse_mapping(self, dictionary):
        nd = dict()
        for key, val in dictionary.items():
            nd[val] = key
        return nd

    def export_to_csv(self, stats, json_file):
        """
        Meant to load the fields and rows into a csv with the given filename
        """
        json_object = json.dumps(stats, indent = 4)
        with open(json_file, "w") as outfile: 
            outfile.write(json_object)

    def import_from_csv(self):
        """
        Meant to load the csv file of interest into a loadable form
        """
        fields = []
        rows = []
        with open(self.label_csv, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            fields = next(csvreader)
            for row in csvreader:
                rows.append(row)
        self.rows = rows
        self.fields = fields

    def dict_rows(self):
        """
        turns rows of lists to a list of dictionaries.
        Each dictionary will map field name to the value associated with it in that row
        """
        dict_rows = []
        this_index_dict = name_to_index(self.fields)
        for row in self.rows:
            nd = dict()
            for name, index in this_index_dict.items():
                nd[name] = row[index]
            dict_rows.append(nd)
        return dict_rows

    def load_json(self, filename):
        """
        Meant to load a json for direct comparison
        """
        with open(filename) as json_file:
            json_dict = json.load(json_file)

        self.json_dict = json_dict

    def img_name_to_dict_rows(self):
        nd = dict()
        for d in self.dict_rows():
            name = d['file'][self.split_index:]
            nd[name] = d
        return nd

    def gen_stats_dict(self):
        d = dict()

        protected_dict = self.label_mapping[self.protected_attribute]
        prediction_dict = self.label_mapping[self.prediction_attribute]

        options = ['c', 'w']

        d[self.protected_attribute] = dict()
        for key in protected_dict.keys():
            d[self.protected_attribute][key] = dict()
            for o in options:
                d[self.protected_attribute][key][o] = 0

        d[self.prediction_attribute] = dict()
        for key in prediction_dict.keys():
            d[self.prediction_attribute][key] = dict()
            for o in options:
                 d[self.prediction_attribute][key][o] = 0

        d['refined'] = dict()
        for key in self.label_mapping["race"].keys():
            d['refined'][key] = dict()
            for nkey in self.label_mapping["gender"].keys():
                d['refined'][key][nkey] = dict()
                for o in options:
                    d['refined'][key][nkey][o] = 0

        # d['incorrect'] = dict()

        d['gender_accuracies_by_race'] = dict()
        for key in self.label_mapping["race"].keys():
            d['gender_accuracies_by_race'][key] = dict()
            for nkey in self.label_mapping["gender"].keys():
                d['gender_accuracies_by_race'][key][nkey] = dict()
                for o in options:
                    d['gender_accuracies_by_race'][key][nkey][o] = 0

        return d

    def update_stats(self, stats_dict, e_protected_key, e_prediction_key, o_prot, o_pred, a_protected_key, a_prediction_key):
        """
        option = c or w for correct or wrong
        """
        stats_dict[self.protected_attribute][e_protected_key][o_prot] += 1
        stats_dict[self.prediction_attribute][e_prediction_key][o_pred] += 1
        
        if self.prediction_attribute == 'race':
            k1 = e_prediction_key
            k2 = e_protected_key
        else:
            k1 = e_protected_key
            k2 = e_prediction_key
        if o_prot == 'w' and o_pred == 'c':
            stats_dict['refined'][k1][k2]['c'] += 1
        else:
            stats_dict['refined'][k1][k2]['w'] += 1

        gender_classification = None
        expected_race = None
        expected_gender = None
        if self.prediction_attribute == 'race':
            gender_classification = o_prot
            expected_gender = e_protected_key
            expected_race = e_prediction_key
        else:
            gender_classification = o_pred
            expected_gender = e_prediction_key
            expected_race = e_protected_key

        if gender_classification == 'c':
            stats_dict['gender_accuracies_by_race'][expected_race][expected_gender]['c'] += 1
        else:
            stats_dict['gender_accuracies_by_race'][expected_race][expected_gender]['w'] += 1

    def compare(self):
        stats = self.gen_stats_dict()
        csv_dict = self.img_name_to_dict_rows()
        protected_dict = self.label_mapping[self.protected_attribute]
        prediction_dict = self.label_mapping[self.prediction_attribute]
        r_protected_dict = self.label_mapping['r_'  + self.protected_attribute]
        r_prediction_dict = self.label_mapping['r_' + self.prediction_attribute]
        for img_name, value in self.json_dict.items():

            protected_expected = csv_dict[img_name][self.protected_attribute]
            predicted_expected = csv_dict[img_name][self.prediction_attribute]

            expected_protected_val = protected_dict[protected_expected]
            expected_predicted_val = prediction_dict[predicted_expected]

            # print("value:", value)

            eval_protected_val = value['privacy_attribute_predicted']
            eval_prediction_val = value['task_attribute_prediction']
            a_protected_key = r_protected_dict[eval_protected_val]
            a_prediction_key = r_prediction_dict[eval_prediction_val]

            o_prot = expected_protected_val == eval_protected_val
            if o_prot:
                o_prot = 'c'
            else:
                o_prot = 'w'
            o_pred = expected_predicted_val == eval_prediction_val
            if o_pred:
                o_pred = 'c'
            else:
                o_pred = 'w'

            self.update_stats(stats, protected_expected, predicted_expected, o_prot, o_pred, a_protected_key, a_prediction_key)
            
        return stats



if __name__ == '__main__':
    db = Database()
    json_path = '/Users/ethangarza/FairFace/fairface-img-margin025-trainval/val_outcome/pruning_network_fairface_resnet18_scratch_split6_ratio0.2_1.json'
    db.load_json(json_path)
    stats = db.compare()
    n_path = json_path[:-5] + "_analysis" + json_path[-5:]
    db.export_to_csv(stats, n_path)

