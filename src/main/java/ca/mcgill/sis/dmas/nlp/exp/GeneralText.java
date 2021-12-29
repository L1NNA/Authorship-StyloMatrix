package ca.mcgill.sis.dmas.nlp.exp;

import java.io.File;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ca.mcgill.sis.dmas.nlp.corpus.parser.Tagger.Language;
import ca.mcgill.sis.dmas.nlp.corpus.preprocess.Preprocessor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Charsets;
import com.google.common.collect.Iterables;

import auc.Confusion;
import auc.ReadList;
import ca.mcgill.sis.dmas.env.StringResources;
import ca.mcgill.sis.dmas.io.LineSequenceWriter;
import ca.mcgill.sis.dmas.io.Lines;
import ca.mcgill.sis.dmas.io.collection.Counter;
import ca.mcgill.sis.dmas.io.collection.EntryPair;
import ca.mcgill.sis.dmas.io.collection.StreamIterable;
import ca.mcgill.sis.dmas.io.file.DmasFileOperations;
import ca.mcgill.sis.dmas.nlp.corpus.Sentence;
import ca.mcgill.sis.dmas.nlp.corpus.parser.NLPUtilsInitializer;
import ca.mcgill.sis.dmas.nlp.corpus.parser.NlpUtilsOpennlp;
import ca.mcgill.sis.dmas.nlp.corpus.parser.NlpUtilsStandford;
import ca.mcgill.sis.dmas.nlp.corpus.parser.SentenceDetector;
import ca.mcgill.sis.dmas.nlp.corpus.parser.Tagger;
import ca.mcgill.sis.dmas.nlp.corpus.parser.Tokenizer;
import ca.mcgill.sis.dmas.nlp.model.astyle.Document;
import ca.mcgill.sis.dmas.nlp.model.astyle.MathUtilities;
import ca.mcgill.sis.dmas.nlp.model.astyle._2_character.LearnerChar2VecEmbedding;
import ca.mcgill.sis.dmas.nlp.model.astyle._2_character.LearnerChar2VecEmbedding.C2VParam;
import ca.mcgill.sis.dmas.nlp.model.astyle._2_character.LearnerChar2VecEmbedding.CEmbedding;

public class GeneralText {
    private static Logger logger = LoggerFactory.getLogger(GeneralText.class);

    public static void main(String[] args) throws Exception {
        File training_folder = new File(args[0] + "/train");
        File testing_folder = new File(args[0] + "/test");
        String method = args[1].toLowerCase().strip();
        String models = args[2];
        NLPUtilsInitializer.initialize(models);
        Language lang = Language.english;
        Tagger tagger = Tagger.getTagger(lang);
        Preprocessor preprocessor = new Preprocessor(Preprocessor.F_ToLowerCase(), Preprocessor.F_SeperatePunctuation(),
                Preprocessor.F_RemoveEtraSpace());

        List<Document> docs_training = LoadDocuments(training_folder, preprocessor, tagger);
        List<Document> docs_testing = LoadDocuments(testing_folder, preprocessor, tagger);

        if (method.equals("char2vec")) {
            logger.info("Training...");
            C2VParam param = new C2VParam();
            param.optm_parallelism = Runtime.getRuntime().availableProcessors();
            param.optm_aphaUpdateInterval = -1;
            MathUtilities.createExpTable();
            LearnerChar2VecEmbedding p2v = new LearnerChar2VecEmbedding(param);
            p2v.debug = false;
            p2v.train(docs_training);
            logger.info("Infering");
            CEmbedding embd_training = p2v.produceDocEmbd();
            CEmbedding embd_testing = p2v.infer(docs_testing);
            ObjectMapper om = new ObjectMapper();
            om.writeValue(new File(args[0] + "/e_train.json"), embd_training.charEmbedding);
            om.writeValue(new File(args[0] + "/e_test.json"), embd_testing.charEmbedding);
        }

    }

    public static List<Document> LoadDocuments(File root, Preprocessor preprocessor, Tagger tagger) {
        SentenceDetector sd = SentenceDetector.newSentenceDetectorOpenNLP();
        logger.info("Processing {}", root.getAbsolutePath());
        Stream<Document> allDocStream = Arrays.stream(root.listFiles()).filter(file -> file.isFile())
                .map(file -> {
                    Document knownDoc = new Document();
                    knownDoc.id = file.getName();
                    ArrayList<String> knLines = new ArrayList<>();
                    try {
                        String line = Lines.readAll(file.getAbsolutePath(), Charsets.UTF_8, false);
                        if (line.startsWith(StringResources.REGEX_UTF8_BOM)) {
                            line = line.substring(1);
                        }
                        String[] sentences = sd.detectSentences(preprocessor.pass(line));
                        knLines.addAll(Arrays.asList(sentences));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                    knownDoc.rawContent = StringResources.JOINER_TOKEN.join(knLines);
                    knLines.stream().map(l -> new Sentence(l, Tokenizer.tokenizerStandford))
                            .forEach(knownDoc.sentences::add);
                    knownDoc.sentences_tags = knownDoc.sentences.stream().map(sent -> {
                        Sentence tSentence = new Sentence();
                        List<String> tags = tagger.tag(sent.tokens);
                        if (sent.tokens.length != tags.size())
                            logger.error("Unmatched size {} vs {}", sent.tokens.length, tags.size());
                        tSentence.tokens = tags.toArray(new String[tags.size()]);
                        return tSentence;
                    }).collect(Collectors.toCollection(ArrayList::new));

                    return knownDoc;
                }).filter(doc -> doc != null);
        return allDocStream.collect(Collectors.toList());
    }
}
