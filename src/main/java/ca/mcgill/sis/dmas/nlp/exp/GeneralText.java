package ca.mcgill.sis.dmas.nlp.exp;

import java.io.File;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ca.mcgill.sis.dmas.nlp.corpus.parser.Tagger.Language;
import ca.mcgill.sis.dmas.nlp.corpus.preprocess.Preprocessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import ca.mcgill.sis.dmas.nlp.model.astyle._1_lexical.LearnerTL2VecEmbedding3;
import ca.mcgill.sis.dmas.nlp.model.astyle._1_lexical.LearnerTL2VecEmbedding.TL2VParam;
import ca.mcgill.sis.dmas.nlp.model.astyle._1_lexical.LearnerTL2VecEmbedding.TLEmbedding;
import ca.mcgill.sis.dmas.nlp.model.astyle._2_character.LearnerChar2VecEmbedding;
import ca.mcgill.sis.dmas.nlp.model.astyle._2_character.LearnerChar2VecEmbedding.C2VParam;
import ca.mcgill.sis.dmas.nlp.model.astyle._2_character.LearnerChar2VecEmbedding.CEmbedding;
import ca.mcgill.sis.dmas.nlp.model.astyle._3_syntactic.LearnerSyn2VecEmbedding2;
import ca.mcgill.sis.dmas.nlp.model.astyle._3_syntactic.LearnerSyn2VecEmbedding.S2VParam;
import ca.mcgill.sis.dmas.nlp.model.astyle._3_syntactic.LearnerSyn2VecEmbedding.SEmbedding;
import ca.mcgill.sis.dmas.nlp.model.astyle._4_stylometricBasic.Stylometric;
import ca.mcgill.sis.dmas.nlp.model.astyle._4_stylometricBasic.Stylometric.StylometricParam;

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
        HashMap<String, Map<String, double[]>> training_embeddings = new HashMap<>();
        HashMap<String, Map<String, double[]>> testing_embeddings = new HashMap<>();

        if (method.equals("char2vec")) {
            logger.info("Training char2vec ...");
            C2VParam param = new C2VParam();
            param.optm_parallelism = Runtime.getRuntime().availableProcessors();
            param.optm_aphaUpdateInterval = -1;
            MathUtilities.createExpTable();
            LearnerChar2VecEmbedding p2v = new LearnerChar2VecEmbedding(param);
            p2v.debug = false;
            p2v.train(docs_training);
            logger.info("Inferring");
            training_embeddings.put(
                "character", 
                MathUtilities.normalize(p2v.produceDocEmbd().charEmbedding)
                );
            testing_embeddings.put(
                "character", 
                MathUtilities.normalize(p2v.infer(docs_testing).charEmbedding)
                );
        } else if (method.equals("tl2vec")) {
            logger.info("Training tl2vec...");
            TL2VParam param = new TL2VParam();
            param.optm_parallelism = Runtime.getRuntime().availableProcessors();
            param.optm_aphaUpdateInterval = -1;
            MathUtilities.createExpTable();
            LearnerTL2VecEmbedding3 p2v = new LearnerTL2VecEmbedding3(param);
            p2v.debug = false;
            p2v.train(docs_training);
            logger.info("Inferring");
            TLEmbedding train_embd = p2v.produceDocEmbdUnnormalized();
            TLEmbedding test_embd = p2v.inferUnnormalized(docs_testing);
            training_embeddings.put(
                "lexical", 
                MathUtilities.normalize(train_embd.lexicEmbedding)
                );
            testing_embeddings.put(
                "lexical", 
                MathUtilities.normalize(test_embd.lexicEmbedding)
                );

            training_embeddings.put(
                "topical", 
                MathUtilities.normalize(train_embd.topicEmbedding)
                );
            testing_embeddings.put(
                "topical", 
                MathUtilities.normalize(test_embd.topicEmbedding)
                );
        }else if (method.equals("pos2vec")){
            logger.info("Training pos2vec...");
            S2VParam param = new S2VParam();
            param.optm_parallelism = Runtime.getRuntime().availableProcessors();
            param.optm_aphaUpdateInterval = -1;
            MathUtilities.createExpTable();
            LearnerSyn2VecEmbedding2 p2v = new LearnerSyn2VecEmbedding2(param);
            p2v.train(docs_training);
            logger.info("Inferring");
            training_embeddings.put(
                "syntatic", 
                MathUtilities.normalize(p2v.produceDocEmbd().synEmbedding)
                );
            testing_embeddings.put(
                "syntatic", 
                MathUtilities.normalize(p2v.infer(docs_testing).synEmbedding)
                );
        }else if (method.equals("stylometric")){
            logger.info("Training pos2vec...");
            StylometricParam param = new StylometricParam();
            MathUtilities.createExpTable();
            Stylometric sty = new Stylometric(param);
            sty.train(docs_training);

            logger.info("Inferring");
            training_embeddings.put(
                "stylometric", 
                MathUtilities.normalize(sty.getDocEmbedding())
                );
            testing_embeddings.put(
                "stylometric", 
                MathUtilities.normalize(sty.inferNewDocEmbedding(docs_testing))
                );
        }

        ObjectMapper om = new ObjectMapper();
        om.writeValue(new File(args[0] + "/e_train.json"), training_embeddings);
        om.writeValue(new File(args[0] + "/e_test.json"), testing_embeddings);
    }

    public static List<Document> LoadDocuments(File root, Preprocessor preprocessor, Tagger tagger) {
        SentenceDetector sd = SentenceDetector.newSentenceDetectorOpenNLP();
        logger.info("Processing {}", root.getAbsolutePath());
        Stream<Document> allDocStream = Arrays.stream(root.listFiles()).filter(file -> file.isFile())
                .map(file -> {
                    Document knownDoc = new Document();
                    knownDoc.id = root.getName() + "_" + file.getName();
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
