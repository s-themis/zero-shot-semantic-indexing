import json
import pathlib
import pandas as pd

from collections import defaultdict


def parse_raw_descrs_data(descriptors_data_bin: str):
    descriptors_data_bin = pathlib.Path(descriptors_data_bin)

    with open(descriptors_data_bin, "r") as f:
        all_descriptors_data = {}
        for line in f:
            line = line.strip()
            if line.startswith("*NEWRECORD"):
                # initialize dict to collect data for current descriptor
                current_descriptor_data = defaultdict(list)

            elif line.startswith("UI = "):
                # data collection for current descriptor completed
                descriptor_ui = line.split("=")[1].strip()
                current_descriptor_data["entry_terms"] = list(map(lambda s: s.split("|")[0], current_descriptor_data["entry_terms"]))
                current_descriptor_data["print_entry_terms"] = list(map(lambda s: s.split("|")[0], current_descriptor_data["print_entry_terms"]))
                current_descriptor_data["all_entry_terms"] = current_descriptor_data["entry_terms"] + current_descriptor_data["print_entry_terms"]
                all_descriptors_data[descriptor_ui] = current_descriptor_data
            
            elif not line:
                # skip blank lines
                pass

            else:
                # parse and append data for current descriptor
                parsed = line.split("=")
                field = parsed[0].strip()
                value = "=".join(parsed[1:]).strip()

                if field == "MH":
                    current_descriptor_data["name"] = value
                elif field == "ENTRY":
                    current_descriptor_data["entry_terms"].append(value)
                elif field == "PRINT ENTRY":
                    current_descriptor_data["print_entry_terms"].append(value)
                elif field == "MS":
                    current_descriptor_data["scope_note"] = value
                
    return all_descriptors_data


def extract_emerging_descrs(use_cases_selected_csv: str, descriptors_data_bin: str):
    use_cases_selected_csv = pathlib.Path(use_cases_selected_csv)
    descriptors_data_bin = pathlib.Path(descriptors_data_bin)

    df = pd.read_csv(use_cases_selected_csv)
    df = df[["Descr. UI", "Descr. Name", "PHs UIs", "PHs", "PHex UIs"]]
    df["PHex UIs"] = df["PHex UIs"].apply(lambda s: s.split("~"))

    all_descriptors_data = parse_raw_descrs_data(descriptors_data_bin)

    df["name"] = df["Descr. UI"].apply(lambda UI: all_descriptors_data[UI]["name"])
    df["entry_terms"] = df["Descr. UI"].apply(lambda UI: all_descriptors_data[UI]["entry_terms"])
    df["print_entry_terms"] = df["Descr. UI"].apply(lambda UI: all_descriptors_data[UI]["print_entry_terms"])
    df["all_entry_terms"] = df["Descr. UI"].apply(lambda UI: all_descriptors_data[UI]["all_entry_terms"])
    df["scope_note"] = df["Descr. UI"].apply(lambda UI: all_descriptors_data[UI]["scope_note"])

    return df

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--emerging_descrs_csv", type=str)
    parser.add_argument("--raw_descrs_bin", type=str)
    parser.add_argument("--dest_json", type=str)
    args = parser.parse_args()

    df = extract_emerging_descrs(args.emerging_descrs_csv, args.raw_descrs_bin)
    with open(args.dest_json, "w") as f:
        f.write(json.dumps({"descriptors": df.to_dict(orient="records")}))
