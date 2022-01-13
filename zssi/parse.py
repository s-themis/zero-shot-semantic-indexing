import spacy


class Parser:
    """Defines the interface that any subclass should implement.

    """
    def __init__(self):
        raise NotImplementedError

    def parse(self, doc):
        """Parses a single document.

        Args:
            doc (dict): The document to parse represented as a dict.

        Returns:
            List(str): The parsed document.
        """
        raise NotImplementedError

    def get_id(self, doc):
        """Extracts the id of the document.

        Args:
            doc (dict): The document represented as a dict.

        Returns:
            int: The id of the document.
        """
        return doc["pmid"]


class SentenceSegmentationParser(Parser):
    """A class to extract the title and the segmented sentences of the abstract.

    """
    def __init__(self):
        """Initializes the parser with the 'en_core_web_sm' language model for sentence segmentation.

        """
        self.nlp = spacy.load('en_core_web_sm')

    def parse(self, doc):
        segmented_sentences = []
        segmented_sentences.append(doc['title'])
        segmented_sentences.extend(
            [str(sent) for sent in self.nlp(doc['abstractText']).sents])
        return segmented_sentences


class WholeTextParser(Parser):
    """A class to extract the title and the abstract as a single piece of text.

    """
    def __init__(self):
        pass

    def parse(self, doc):
        whole_text = []
        whole_text.append(doc['title'] + " " + doc['abstractText'])
        return whole_text


class AugmentedDescriptorParser(Parser):
    def __init__(self):
        pass

    def parse(self, doc):
        return [doc["label"]]

    def get_id(self, doc):
        return doc["UI"]
