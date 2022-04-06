# activate venv
source venv/bin/activate

# create required directories
mkdir datastores
mkdir datastores/mesh_zssi_2006

# convert documents to jsonl for stream processing
python zssi/json_to_jsonl.py --input_json data/external/test_docs_2006.json --output_jsonl datastores/mesh_zssi_2006/test_docs_2006.jsonl --objects_field documents --limit 10000000

# extract emerging decriptors and corresponding data
python zssi/extract_emerging_descrs.py --emerging_descrs_csv data/external/UseCasesSelected_2006_extended.csv --raw_descrs_bin data/external/d2022.bin --dest_json data/intermediate/emerging_descrs_2006.json

# convert descriptors to jsonl for stream processing
python zssi/json_to_jsonl.py --input_json data/intermediate/emerging_descrs_2006.json --output_jsonl data/intermediate/emerging_descrs_2006.jsonl --objects_field descriptors --limit 1000

# generate all descriptor variation with extra info
python zssi/generate_descrs_variations.py --descrs_jsonl data/intermediate/emerging_descrs_2006.jsonl --dest_dir datastores/mesh_zssi_2006
