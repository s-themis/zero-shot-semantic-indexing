import jsonlines as jsonl


class DescriptorAugmenter:
    def __init__(self, descriptor_objs, output_name):
        self.descriptor_objs = descriptor_objs
        self.output_name = output_name

    def _augment(self, entry_terms=False, scope_note=False, seperator=" "):
        augmented_descriptors = []
        for descr_obj in self.descriptor_objs:
            aug_descr = descr_obj["name"]
            if entry_terms:
                if descr_obj["entry_terms"]:
                    aug_descr += " " + " ".join(descr_obj["entry_terms"])
                else:
                    continue
            if scope_note:
                if descr_obj["scope_note"]:
                    aug_descr += " " + descr_obj["scope_note"]
                else:
                    continue
            augmented_descriptors.append({
                "UI": descr_obj["UI"],
                "label": aug_descr
            })
        return augmented_descriptors

    def save_all_flavors(self):
        flavors = [{
            "entry_terms": False,
            "scope_note": False,
            "partial_filename": "name"
        }, {
            "entry_terms": True,
            "scope_note": False,
            "partial_filename": "name_entry_terms"
        }, {
            "entry_terms": False,
            "scope_note": True,
            "partial_filename": "name_scope_note"
        }, {
            "entry_terms": True,
            "scope_note": True,
            "partial_filename": "name_entry_terms_scope_note"
        }]

        for flavor in flavors:
            entry_terms, scope_note, part_fname = flavor.values()
            filename = "augmented_descriptors/" + self.output_name + "_" + part_fname + ".jsonl"
            with jsonl.open(filename, mode='w') as f:
                f.write_all(
                    self._augment(entry_terms=entry_terms,
                                  scope_note=scope_note))
