import pytest

from zssi.datagen import JsonlDocGenerator
from zssi.parse import SentenceSegmentationDocParser, WholeTextDocParser


@pytest.fixture
def sample_doc():
    return {
        "abstractText":
        "DNA methylation is an epigenetic alteration that may lead to carcinogenesis by silencing key tumor suppressor genes. Hypermethylation of the paired box gene 1 (PAX1) promoter is important in cervical cancer development. Here, PAX1 methylation levels were compared between Uyghur and Han patients with cervical lesions. Data on PAX1 methylation in different cervical lesions were obtained from the Gene Expression Omnibus (GEO) database, whereas data on survival and PAX1 mRNA expression in invasive cervical cancer (ICC) were retrieved from the Cancer Genome Atlas (TCGA) database. MassARRAY spectrometry was used to detect methylation of 19 CpG sites in the promoter region of PAX1, whereas gene mass spectrograms were drawn by Matrix-Assisted Laser Desorption/Ionization Time-of-Flight Mass Spectrometry. Human papillomavirus (HPV) 16 infection was detected by polymerase chain reaction. PAX1 methylation in high-grade squamous intraepithelial lesion (HSIL) and ICC was significantly higher than in normal tissues. PAX1 hypermethylation was associated with poor prognosis and reduced transcription. ICC-specific PAX1 promoter methylation involved distinct CpG sites in Uyghur and Han patients HPV16 infection in HSIL and ICC patient was significantly higher than in normal women (p < 0.05). Our study revealed a strong association between PAX1 methylation and the development of cervical cancer. Moreover, hypermethylation of distinct CpG sites may induce HSIL transformation into ICC in both Uyghur and Han patients. Our results suggest the existence of ethnic differences in the genetic susceptibility to cervical cancer. Finally, PAX1 methylation and HPV infection exhibited synergistic effects on cervical carcinogenesis.",
        "pmid":
        31589957,
        "title":
        "Association between dense PAX1 promoter methylation and HPV16 infection in cervical squamous epithelial neoplasms of Xin Jiang Uyghur and Han women."
    }


@pytest.fixture
def sample_doc_whole_text():
    return [
        "Association between dense PAX1 promoter methylation and HPV16 infection in cervical squamous epithelial neoplasms of Xin Jiang Uyghur and Han women. DNA methylation is an epigenetic alteration that may lead to carcinogenesis by silencing key tumor suppressor genes. Hypermethylation of the paired box gene 1 (PAX1) promoter is important in cervical cancer development. Here, PAX1 methylation levels were compared between Uyghur and Han patients with cervical lesions. Data on PAX1 methylation in different cervical lesions were obtained from the Gene Expression Omnibus (GEO) database, whereas data on survival and PAX1 mRNA expression in invasive cervical cancer (ICC) were retrieved from the Cancer Genome Atlas (TCGA) database. MassARRAY spectrometry was used to detect methylation of 19 CpG sites in the promoter region of PAX1, whereas gene mass spectrograms were drawn by Matrix-Assisted Laser Desorption/Ionization Time-of-Flight Mass Spectrometry. Human papillomavirus (HPV) 16 infection was detected by polymerase chain reaction. PAX1 methylation in high-grade squamous intraepithelial lesion (HSIL) and ICC was significantly higher than in normal tissues. PAX1 hypermethylation was associated with poor prognosis and reduced transcription. ICC-specific PAX1 promoter methylation involved distinct CpG sites in Uyghur and Han patients HPV16 infection in HSIL and ICC patient was significantly higher than in normal women (p < 0.05). Our study revealed a strong association between PAX1 methylation and the development of cervical cancer. Moreover, hypermethylation of distinct CpG sites may induce HSIL transformation into ICC in both Uyghur and Han patients. Our results suggest the existence of ethnic differences in the genetic susceptibility to cervical cancer. Finally, PAX1 methylation and HPV infection exhibited synergistic effects on cervical carcinogenesis."
    ]


@pytest.fixture
def sample_doc_segmented_sentences():
    return [
        "Association between dense PAX1 promoter methylation and HPV16 infection in cervical squamous epithelial neoplasms of Xin Jiang Uyghur and Han women.",
        "DNA methylation is an epigenetic alteration that may lead to carcinogenesis by silencing key tumor suppressor genes.",
        "Hypermethylation of the paired box gene 1 (PAX1) promoter is important in cervical cancer development.",
        "Here, PAX1 methylation levels were compared between Uyghur and Han patients with cervical lesions.",
        "Data on PAX1 methylation in different cervical lesions were obtained from the Gene Expression Omnibus (GEO) database, whereas data on survival and PAX1 mRNA expression in invasive cervical cancer (ICC) were retrieved from the Cancer Genome Atlas (TCGA) database.",
        "MassARRAY spectrometry was used to detect methylation of 19 CpG sites in the promoter region of PAX1, whereas gene mass spectrograms were drawn by Matrix-Assisted Laser Desorption/Ionization Time-of-Flight Mass Spectrometry.",
        "Human papillomavirus (HPV) 16 infection was detected by polymerase chain reaction.",
        "PAX1 methylation in high-grade squamous intraepithelial lesion (HSIL) and ICC was significantly higher than in normal tissues.",
        "PAX1 hypermethylation was associated with poor prognosis and reduced transcription.",
        "ICC-specific PAX1 promoter methylation involved distinct CpG sites in Uyghur and Han patients HPV16 infection in HSIL and ICC patient was significantly higher than in normal women (p < 0.05).",
        "Our study revealed a strong association between PAX1 methylation and the development of cervical cancer.",
        "Moreover, hypermethylation of distinct CpG sites may induce HSIL transformation into ICC in both Uyghur and Han patients.",
        "Our results suggest the existence of ethnic differences in the genetic susceptibility to cervical cancer.",
        "Finally, PAX1 methylation and HPV infection exhibited synergistic effects on cervical carcinogenesis.",
    ]


def sample_2_docs_flat_batches_segmented_sentences():
    return [
        "Association between dense PAX1 promoter methylation and HPV16 infection in cervical squamous epithelial neoplasms of Xin Jiang Uyghur and Han women.",
        "DNA methylation is an epigenetic alteration that may lead to carcinogenesis by silencing key tumor suppressor genes.",
        "Hypermethylation of the paired box gene 1 (PAX1) promoter is important in cervical cancer development.",
        "Here, PAX1 methylation levels were compared between Uyghur and Han patients with cervical lesions.",
        "Data on PAX1 methylation in different cervical lesions were obtained from the Gene Expression Omnibus (GEO) database, whereas data on survival and PAX1 mRNA expression in invasive cervical cancer (ICC) were retrieved from the Cancer Genome Atlas (TCGA) database.",
        "MassARRAY spectrometry was used to detect methylation of 19 CpG sites in the promoter region of PAX1, whereas gene mass spectrograms were drawn by Matrix-Assisted Laser Desorption/Ionization Time-of-Flight Mass Spectrometry.",
        "Human papillomavirus (HPV) 16 infection was detected by polymerase chain reaction.",
        "PAX1 methylation in high-grade squamous intraepithelial lesion (HSIL) and ICC was significantly higher than in normal tissues.",
        "PAX1 hypermethylation was associated with poor prognosis and reduced transcription.",
        "ICC-specific PAX1 promoter methylation involved distinct CpG sites in Uyghur and Han patients HPV16 infection in HSIL and ICC patient was significantly higher than in normal women (p < 0.05).",
        "Our study revealed a strong association between PAX1 methylation and the development of cervical cancer.",
        "Moreover, hypermethylation of distinct CpG sites may induce HSIL transformation into ICC in both Uyghur and Han patients.",
        "Our results suggest the existence of ethnic differences in the genetic susceptibility to cervical cancer.",
        "Finally, PAX1 methylation and HPV infection exhibited synergistic effects on cervical carcinogenesis.",
        "Unity and diversity among viral kinases.",
        "Viral kinases are known to undergo autophosphorylation and also phosphorylate viral and host substrates.",
        "Viral kinases have been implicated in various diseases and are also known to acquire host kinases for mimicking cellular functions and exhibit virulence.",
        "Although substantial analyses have been reported in the literature on diversity of viral kinases, there is a gap in the understanding of sequence and structural similarity among kinases from different classes of viruses.",
        "In this study, we performed a comprehensive analysis of protein kinases encoded in viral genomes.",
        "Homology search methods have been used to identify kinases from 104,282 viral genomic datasets.",
        "Serine/threonine and tyrosine kinases are identified only in 390 viral genomes.",
        "Out of seven viral classes that are based on nature of genetic material, only viruses having double-stranded DNA and single-stranded RNA retroviruses are found to encode kinases.",
        "The 716 identified protein kinases are classified into 63 subfamilies based on their sequence similarity within each cluster, and sequence signatures have been identified for each subfamily.",
        "11 clusters are well represented with at least 10 members in each of these clusters.",
        "Kinases from dsDNA viruses, Phycodnaviridae which infect green algae and Herpesvirales that infect vertebrates including human, form a major group.",
        "From our analysis, it has been observed that the protein kinases in viruses belonging to same taxonomic lineages form discrete clusters and the kinases encoded in alphaherpesvirus form host-specific clusters.",
        "A comprehensive sequence and structure-based analysis enabled us to identify the conserved residues or motifs in kinase catalytic domain regions across all viral kinases.",
        "Conserved sequence regions that are specific to a particular viral kinase cluster and the kinases that show close similarity to eukaryotic kinases were identified by using sequence and three-dimensional structural regions of eukaryotic kinases as reference.",
        "The regions specific to each viral kinase cluster can be used as signatures in the future in classifying uncharacterized viral kinases.",
        "We note that kinases from giant viruses Marseilleviridae have close similarity to viral oncogenes in the functional regions and in putative substrate binding regions indicating their possible role in cancer.",
    ]


def sample_2_docs_flat_batches_segmented_sentences_doc_ids():
    return [
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589957,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
        31589960,
    ]


def sample_2_docs_flat_batches_whole_text():
    return [
        "Association between dense PAX1 promoter methylation and HPV16 infection in cervical squamous epithelial neoplasms of Xin Jiang Uyghur and Han women. DNA methylation is an epigenetic alteration that may lead to carcinogenesis by silencing key tumor suppressor genes. Hypermethylation of the paired box gene 1 (PAX1) promoter is important in cervical cancer development. Here, PAX1 methylation levels were compared between Uyghur and Han patients with cervical lesions. Data on PAX1 methylation in different cervical lesions were obtained from the Gene Expression Omnibus (GEO) database, whereas data on survival and PAX1 mRNA expression in invasive cervical cancer (ICC) were retrieved from the Cancer Genome Atlas (TCGA) database. MassARRAY spectrometry was used to detect methylation of 19 CpG sites in the promoter region of PAX1, whereas gene mass spectrograms were drawn by Matrix-Assisted Laser Desorption/Ionization Time-of-Flight Mass Spectrometry. Human papillomavirus (HPV) 16 infection was detected by polymerase chain reaction. PAX1 methylation in high-grade squamous intraepithelial lesion (HSIL) and ICC was significantly higher than in normal tissues. PAX1 hypermethylation was associated with poor prognosis and reduced transcription. ICC-specific PAX1 promoter methylation involved distinct CpG sites in Uyghur and Han patients HPV16 infection in HSIL and ICC patient was significantly higher than in normal women (p < 0.05). Our study revealed a strong association between PAX1 methylation and the development of cervical cancer. Moreover, hypermethylation of distinct CpG sites may induce HSIL transformation into ICC in both Uyghur and Han patients. Our results suggest the existence of ethnic differences in the genetic susceptibility to cervical cancer. Finally, PAX1 methylation and HPV infection exhibited synergistic effects on cervical carcinogenesis.",
        "Unity and diversity among viral kinases. Viral kinases are known to undergo autophosphorylation and also phosphorylate viral and host substrates. Viral kinases have been implicated in various diseases and are also known to acquire host kinases for mimicking cellular functions and exhibit virulence. Although substantial analyses have been reported in the literature on diversity of viral kinases, there is a gap in the understanding of sequence and structural similarity among kinases from different classes of viruses. In this study, we performed a comprehensive analysis of protein kinases encoded in viral genomes. Homology search methods have been used to identify kinases from 104,282 viral genomic datasets. Serine/threonine and tyrosine kinases are identified only in 390 viral genomes. Out of seven viral classes that are based on nature of genetic material, only viruses having double-stranded DNA and single-stranded RNA retroviruses are found to encode kinases. The 716 identified protein kinases are classified into 63 subfamilies based on their sequence similarity within each cluster, and sequence signatures have been identified for each subfamily. 11 clusters are well represented with at least 10 members in each of these clusters. Kinases from dsDNA viruses, Phycodnaviridae which infect green algae and Herpesvirales that infect vertebrates including human, form a major group. From our analysis, it has been observed that the protein kinases in viruses belonging to same taxonomic lineages form discrete clusters and the kinases encoded in alphaherpesvirus form host-specific clusters. A comprehensive sequence and structure-based analysis enabled us to identify the conserved residues or motifs in kinase catalytic domain regions across all viral kinases. Conserved sequence regions that are specific to a particular viral kinase cluster and the kinases that show close similarity to eukaryotic kinases were identified by using sequence and three-dimensional structural regions of eukaryotic kinases as reference. The regions specific to each viral kinase cluster can be used as signatures in the future in classifying uncharacterized viral kinases. We note that kinases from giant viruses Marseilleviridae have close similarity to viral oncogenes in the functional regions and in putative substrate binding regions indicating their possible role in cancer.",
    ]


def sample_2_docs_flat_batches_whole_text_doc_ids():
    return [
        31589957,
        31589960,
    ]


class TestDocParsing:
    def test_sentence_segmentation_doc_parser_parsing(
            self, sample_doc, sample_doc_segmented_sentences):
        parser = SentenceSegmentationDocParser()
        parsed_doc = parser.parse(sample_doc)
        assert parsed_doc == sample_doc_segmented_sentences

    def test_sentence_segmentation_doc_parser_doc_id(self, sample_doc):
        parser = SentenceSegmentationDocParser()
        doc_id = parser.get_id(sample_doc)
        assert doc_id == sample_doc["pmid"]

    def test_whole_text_doc_parser_parsing(self, sample_doc,
                                           sample_doc_whole_text):
        parser = WholeTextDocParser()
        parsed_doc = parser.parse(sample_doc)
        assert parsed_doc == sample_doc_whole_text

    def test_whole_text_doc_parser_parsing_doc_id(self, sample_doc):
        parser = WholeTextDocParser()
        doc_id = parser.get_id(sample_doc)
        assert doc_id == sample_doc["pmid"]


class TestDocDatagen:
    def nest_flat_batches(self, flat_batches, batch_size):
        nested_batches = []
        batch = []
        for example in flat_batches:
            batch.append(example)
            if len(batch) == batch_size:
                nested_batches.append(batch)
                batch = []
        if batch:
            nested_batches.append(batch)
        return nested_batches

    @pytest.mark.parametrize(
        "doc_parser,batch_size,flat_batches,flat_batches_doc_ids", [
            (SentenceSegmentationDocParser(), 3,
             sample_2_docs_flat_batches_segmented_sentences(),
             sample_2_docs_flat_batches_segmented_sentences_doc_ids()),
            (SentenceSegmentationDocParser(), 5,
             sample_2_docs_flat_batches_segmented_sentences(),
             sample_2_docs_flat_batches_segmented_sentences_doc_ids()),
            (SentenceSegmentationDocParser(), 10,
             sample_2_docs_flat_batches_segmented_sentences(),
             sample_2_docs_flat_batches_segmented_sentences_doc_ids()),
            (WholeTextDocParser(), 1, sample_2_docs_flat_batches_whole_text(),
             sample_2_docs_flat_batches_whole_text_doc_ids()),
            (WholeTextDocParser(), 2, sample_2_docs_flat_batches_whole_text(),
             sample_2_docs_flat_batches_whole_text_doc_ids()),
        ])
    def test_jsonl_doc_datagen(self, doc_parser, batch_size, flat_batches,
                               flat_batches_doc_ids):
        path_to_jsonl = 'test_data/test_2006_text_only_2_docs.jsonl'
        datagen = JsonlDocGenerator(path_to_jsonl=path_to_jsonl,
                                    doc_parser=doc_parser)
        doc_generator = datagen.generate(batch_size=batch_size)
        batches = []
        batches_doc_ids = []
        for batch, batch_doc_ids in doc_generator:
            batches.append(batch)
            batches_doc_ids.append(batch_doc_ids)
        assert batches == self.nest_flat_batches(
            flat_batches,
            batch_size) and batches_doc_ids == self.nest_flat_batches(
                flat_batches_doc_ids, batch_size)
