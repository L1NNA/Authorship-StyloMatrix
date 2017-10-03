package ca.mcgill.sis.dmas.nlp.corpus.parser;

import java.util.Arrays;
import java.util.HashSet;

import org.apache.commons.lang3.NotImplementedException;

import ca.mcgill.sis.dmas.nlp.corpus.parser.Tagger.Language;

public class StopWords {

	public static HashSet<String> getStopWords(Language language) {
		switch (language) {
		case english:
			return stopWordsEn;
		case greek:
			return stopWordsGreek;
		case spanish:
			return stopWordsSp;
		case dutch:
			return stopWordsDu;
		default:
			throw new NotImplementedException("The request stopword for " + language + " is not implemented.");
		}
	}

	public static final HashSet<String> stopWordsGreek = new HashSet<>(Arrays.asList("Α∆ΙΑΚΟΠΑ", "ΑΙ", "ΑΚΟΜΑ", "ΑΚΟΜΗ",
			"ΑΚΡΙΒΩΣ", "ΑΛΗΘΕΙΑ", "ΑΛΗΘΙ�?Α", "ΑΛΛΑ", "ΑΛΛΑΧΟΥ", "ΑΛΛΕΣ", "ΑΛΛΗ", "ΑΛΛΗ�?", "ΑΛΛΗΣ", "ΑΛΛΙΩΣ",
			"ΑΛΛΙΩΤΙΚΑ", "ΑΛΛΟ", "ΑΛΛΟΙ", "ΑΛΛΟΙΩΣ", "ΑΛΛΟΙΩΤΙΚΑ", "ΑΛΛΟ�?", "ΑΛΛΟΣ", "ΑΛΛΟΤΕ", "ΑΛΛΟΥ", "ΑΛΛΟΥΣ",
			"ΑΛΛΩ�?", "ΑΜΑ", "ΑΜΕΣΑ", "ΑΜΕΣΩΣ", "Α�?", "Α�?Α", "Α�?ΑΜΕΣΑ", "Α�?ΑΜΕΤΑΞΥ", "Α�?ΕΥ", "Α�?ΤΙ", "Α�?ΤΙΠΕΡΑ", "Α�?ΤΙΣ",
			"Α�?Ω", "Α�?ΩΤΕΡΩ", "ΑΞΑΦ�?Α", "ΑΠ", "ΑΠΕ�?Α�?ΤΙ", "ΑΠΟ", "ΑΠΟΨΕ", "ΑΡΑ", "ΑΡΑΓΕ", "ΑΡΓΑ", "ΑΡΓΟΤΕΡΟ",
			"ΑΡΙΣΤΕΡΑ", "ΑΡΚΕΤΑ", "ΑΡΧΙΚΑ", "ΑΣ", "ΑΥΡΙΟ", "ΑΥΤΑ", "ΑΥΤΕΣ", "ΑΥΤΗ", "ΑΥΤΗ�?", "ΑΥΤΗΣ", "ΑΥΤΟ", "ΑΥΤΟΙ",
			"ΑΥΤΟ�?", "ΑΥΤΟΣ", "ΑΥΤΟΥ", "ΑΥΤΟΥΣ", "ΑΥΤΩ�?", "ΑΦΟΤΟΥ", "ΑΦΟΥ", "ΒΕΒΑΙΑ", "ΒΕΒΑΙΟΤΑΤΑ", "ΓΙ", "ΓΙΑ",
			"ΓΡΗΓΟΡΑ", "ΓΥΡΩ", "∆Α", "∆Ε", "∆ΕΙ�?Α", "∆Ε�?", "∆ΕΞΙΑ", "∆ΗΘΕ�?", "∆ΗΛΑ∆Η", "∆Ι", "∆ΙΑ", "∆ΙΑΡΚΩΣ", "∆ΙΚΑ",
			"∆ΙΚΟ", "∆ΙΚΟΙ", "∆ΙΚΟΣ", "∆ΙΚΟΥ", "∆ΙΚΟΥΣ", "∆ΙΟΛΟΥ", "∆ΙΠΛΑ", "∆ΙΧΩΣ", "ΕΑ�?", "ΕΑΥΤΟ", "ΕΑΥΤΟ�?", "ΕΑΥΤΟΥ",
			"ΕΑΥΤΟΥΣ", "ΕΑΥΤΩ�?", "ΕΓΚΑΙΡΑ", "ΕΓΚΑΙΡΩΣ", "ΕΓΩ", "Ε∆Ω", "ΕΙ∆ΕΜΗ", "ΕΙΘΕ", "ΕΙΜΑΙ", "ΕΙΜΑΣΤΕ", "ΕΙ�?ΑΙ",
			"ΕΙΣ", "ΕΙΣΑΙ", "ΕΙΣΑΣΤΕ", "ΕΙΣΤΕ", "ΕΙΤΕ", "ΕΙΧΑ", "ΕΙΧΑΜΕ", "ΕΙΧΑ�?", "ΕΙΧΑΤΕ", "ΕΙΧΕ", "ΕΙΧΕΣ", "ΕΚΑΣΤΑ",
			"ΕΚΑΣΤΕΣ", "ΕΚΑΣΤΗ", "ΕΚΑΣΤΗ�?", "ΕΚΑΣΤΗΣ", "ΕΚΑΣΤΟ", "ΕΚΑΣΤΟΙ", "ΕΚΑΣΤΟ�?", "ΕΚΑΣΤΟΣ", "ΕΚΑΣΤΟΥ", "ΕΚΑΣΤΟΥΣ",
			"ΕΚΑΣΤΩ�?", "ΕΚΕΙ", "ΕΚΕΙ�?Α", "ΕΚΕΙ�?ΕΣ", "ΕΚΕΙ�?Η", "ΕΚΕΙ�?Η�?", "ΕΚΕΙ�?ΗΣ", "ΕΚΕΙ�?Ο", "ΕΚΕΙ�?ΟΙ", "ΕΚΕΙ�?Ο�?",
			"ΕΚΕΙ�?ΟΣ", "ΕΚΕΙ�?ΟΥ", "ΕΚΕΙ�?ΟΥΣ", "ΕΚΕΙ�?Ω�?", "ΕΚΤΟΣ", "ΕΜΑΣ", "ΕΜΕΙΣ", "ΕΜΕ�?Α", "ΕΜΠΡΟΣ", "Ε�?", "Ε�?Α",
			"Ε�?Α�?", "Ε�?ΑΣ", "Ε�?ΟΣ", "Ε�?ΤΕΛΩΣ", "Ε�?ΤΟΣ", "Ε�?ΤΩΜΕΤΑΞΥ", "Ε�?Ω", "ΕΞ", "ΕΞΑΦ�?Α", "ΕΞΗΣ", "ΕΞΙΣΟΥ", "ΕΞΩ",
			"ΕΠΑ�?Ω", "ΕΠΕΙ∆Η", "ΕΠΕΙΤΑ", "ΕΠΙ", "ΕΠΙΣΗΣ", "ΕΠΟΜΕ�?ΩΣ", "ΕΣΑΣ", "ΕΣΕΙΣ", "ΕΣΕ�?Α", "ΕΣΤΩ", "ΕΣΥ", "ΕΤΕΡΑ",
			"ΕΤΕΡΑΙ", "ΕΤΕΡΑΣ", "ΕΤΕΡΕΣ", "ΕΤΕΡΗ", "ΕΤΕΡΗΣ", "ΕΤΕΡΟ", "ΕΤΕΡΟΙ", "ΕΤΕΡΟ�?", "ΕΤΕΡΟΣ", "ΕΤΕΡΟΥ", "ΕΤΕΡΟΥΣ",
			"ΕΤΕΡΩ�?", "ΕΤΟΥΤΑ", "ΕΤΟΥΤΕΣ", "ΕΤΟΥΤΗ", "ΕΤΟΥΤΗ�?", "ΕΤΟΥΤΗΣ", "ΕΤΟΥΤΟ", "ΕΤΟΥΤΟΙ", "ΕΤΟΥΤΟ�?", "ΕΤΟΥΤΟΣ",
			"ΕΤΟΥΤΟΥ", "ΕΤΟΥΤΟΥΣ", "ΕΤΟΥΤΩ�?", "ΕΤΣΙ", "ΕΥΓΕ", "ΕΥΘΥΣ", "ΕΥΤΥΧΩΣ", "ΕΦΕΞΗΣ", "ΕΧΕΙ", "ΕΧΕΙΣ", "ΕΧΕΤΕ",
			"ΕΧΘΕΣ", "ΕΧΟΜΕ", "ΕΧΟΥΜΕ", "ΕΧΟΥ�?", "ΕΧΤΕΣ", "ΕΧΩ", "ΕΩΣ", "Η", "Η∆Η", "ΗΜΑΣΤΑ�?", "ΗΜΑΣΤΕ", "ΗΜΟΥ�?",
			"ΗΣΑΣΤΑ�?", "ΗΣΑΣΤΕ", "ΗΣΟΥ�?", "ΗΤΑ�?", "ΗΤΑ�?Ε", "ΗΤΟΙ", "ΗΤΤΟ�?", "ΘΑ", "Ι", "Ι∆ΙΑ", "Ι∆ΙΑ�?", "Ι∆ΙΑΣ",
			"Ι∆ΙΕΣ", "Ι∆ΙΟ", "Ι∆ΙΟΙ", "Ι∆ΙΟ�?", "Ι∆ΙΟΣ", "Ι∆ΙΟΥ", "Ι∆ΙΟΥΣ", "Ι∆ΙΩ�?", "Ι∆ΙΩΣ", "ΙΙ", "ΙΙΙ", "ΙΣΑΜΕ",
			"ΙΣΙΑ", "ΙΣΩΣ", "ΚΑΘΕ", "ΚΑΘΕΜΙΑ", "ΚΑΘΕΜΙΑΣ", "ΚΑΘΕ�?Α", "ΚΑΘΕ�?ΑΣ", "ΚΑΘΕ�?ΟΣ", "ΚΑΘΕΤΙ", "ΚΑΘΟΛΟΥ", "ΚΑΘΩΣ",
			"ΚΑΙ", "ΚΑΚΑ", "ΚΑΚΩΣ", "ΚΑΛΑ", "ΚΑΛΩΣ", "ΚΑΜΙΑ", "ΚΑΜΙΑ�?", "ΚΑΜΙΑΣ", "ΚΑΜΠΟΣΑ", "ΚΑΜΠΟΣΕΣ", "ΚΑΜΠΟΣΗ",
			"ΚΑΜΠΟΣΗ�?", "ΚΑΜΠΟΣΗΣ", "ΚΑΜΠΟΣΟ", "ΚΑΜΠΟΣΟΙ", "ΚΑΜΠΟΣΟ�?", "ΚΑΜΠΟΣΟΣ", "ΚΑΜΠΟΣΟΥ", "ΚΑΜΠΟΣΟΥΣ", "ΚΑΜΠΟΣΩ�?",
			"ΚΑ�?ΕΙΣ", "ΚΑ�?Ε�?", "ΚΑ�?Ε�?Α", "ΚΑ�?Ε�?Α�?", "ΚΑ�?Ε�?ΑΣ", "ΚΑ�?Ε�?ΟΣ", "ΚΑΠΟΙΑ", "ΚΑΠΟΙΑ�?", "ΚΑΠΟΙΑΣ", "ΚΑΠΟΙΕΣ",
			"ΚΑΠΟΙΟ", "ΚΑΠΟΙΟΙ", "ΚΑΠΟΙΟ�?", "ΚΑΠΟΙΟΣ", "ΚΑΠΟΙΟΥ", "ΚΑΠΟΙΟΥΣ", "ΚΑΠΟΙΩ�?", "ΚΑΠΟΤΕ", "ΚΑΠΟΥ", "ΚΑΠΩΣ",
			"ΚΑΤ", "ΚΑΤΑ", "ΚΑΤΙ", "ΚΑΤΙΤΙ", "ΚΑΤΟΠΙ�?", "ΚΑΤΩ", "ΚΙΟΛΑΣ", "ΚΛΠ", "ΚΟ�?ΤΑ", "ΚΤΛ", "ΚΥΡΙΩΣ", "ΛΙΓΑΚΙ",
			"ΛΙΓΟ", "ΛΙΓΩΤΕΡΟ", "ΛΟΓΩ", "ΛΟΙΠΑ", "ΛΟΙΠΟ�?", "ΜΑ", "ΜΑΖΙ", "ΜΑΚΑΡΙ", "ΜΑΚΡΥΑ", "ΜΑΛΙΣΤΑ", "ΜΑΛΛΟ�?", "ΜΑΣ",
			"ΜΕ", "ΜΕΘΑΥΡΙΟ", "ΜΕΙΟ�?", "ΜΕΛΕΙ", "ΜΕΛΛΕΤΑΙ", "ΜΕΜΙΑΣ", "ΜΕ�?", "ΜΕΡΙΚΑ", "ΜΕΡΙΚΕΣ", "ΜΕΡΙΚΟΙ", "ΜΕΡΙΚΟΥΣ",
			"ΜΕΡΙΚΩ�?", "ΜΕΣΑ", "ΜΕΤ", "ΜΕΤΑ", "ΜΕΤΑΞΥ", "ΜΕΧΡΙ", "ΜΗ", "ΜΗ∆Ε", "ΜΗ�?", "ΜΗΠΩΣ", "ΜΗΤΕ", "ΜΙΑ", "ΜΙΑ�?",
			"ΜΙΑΣ", "ΜΟΛΙΣ", "ΜΟΛΟ�?ΟΤΙ", "ΜΟ�?ΑΧΑ", "ΜΟ�?ΕΣ", "ΜΟ�?Η", "ΜΟ�?Η�?", "ΜΟ�?ΗΣ", "ΜΟ�?Ο", "ΜΟ�?ΟΙ", "ΜΟ�?ΟΜΙΑΣ",
			"ΜΟ�?ΟΣ", "ΜΟ�?ΟΥ", "ΜΟ�?ΟΥΣ", "ΜΟ�?Ω�?", "ΜΟΥ", "ΜΠΟΡΕΙ", "ΜΠΟΡΟΥ�?", "ΜΠΡΑΒΟ", "ΜΠΡΟΣ", "�?Α", "�?ΑΙ", "�?ΩΡΙΣ",
			"ΞΑ�?Α", "ΞΑΦ�?ΙΚΑ", "Ο", "ΟΙ", "ΟΛΑ", "ΟΛΕΣ", "ΟΛΗ", "ΟΛΗ�?", "ΟΛΗΣ", "ΟΛΟ", "ΟΛΟΓΥΡΑ", "ΟΛΟΙ", "ΟΛΟ�?",
			"ΟΛΟ�?Ε�?", "ΟΛΟΣ", "ΟΛΟΤΕΛΑ", "ΟΛΟΥ", "ΟΛΟΥΣ", "ΟΛΩ�?", "ΟΛΩΣ", "ΟΛΩΣ∆ΙΟΛΟΥ", "ΟΜΩΣ", "ΟΠΟΙΑ", "ΟΠΟΙΑ∆ΗΠΟΤΕ",
			"ΟΠΟΙΑ�?", "ΟΠΟΙΑ�?∆ΗΠΟΤΕ", "ΟΠΟΙΑΣ", "ΟΠΟΙΑΣ∆ΗΠΟΤΕ", "ΟΠΟΙ∆ΗΠΟΤΕ", "ΟΠΟΙΕΣ", "ΟΠΟΙΕΣ∆ΗΠΟΤΕ", "ΟΠΟΙΟ",
			"ΟΠΟΙΟ∆ΗΠΟΤΕ", "ΟΠΟΙΟΙ", "ΟΠΟΙΟ�?", "ΟΠΟΙΟ�?∆ΗΠΟΤΕ", "ΟΠΟΙΟΣ", "ΟΠΟΙΟΣ∆ΗΠΟΤΕ", "ΟΠΟΙΟΥ", "ΟΠΟΙΟΥ∆ΗΠΟΤΕ",
			"ΟΠΟΙΟΥΣ", "ΟΠΟΙΟΥΣ∆ΗΠΟΤΕ", "ΟΠΟΙΩ�?", "ΟΠΟΙΩ�?∆ΗΠΟΤΕ", "ΟΠΟΤΕ", "ΟΠΟΤΕ∆ΗΠΟΤΕ", "ΟΠΟΥ", "ΟΠΟΥ∆ΗΠΟΤΕ", "ΟΠΩΣ",
			"ΟΡΙΣΜΕ�?Α", "ΟΡΙΣΜΕ�?ΕΣ", "ΟΡΙΣΜΕ�?Ω�?", "ΟΡΙΣΜΕ�?ΩΣ", "ΟΣΑ", "ΟΣΑ∆ΗΠΟΤΕ", "ΟΣΕΣ", "ΟΣΕΣ∆ΗΠΟΤΕ", "ΟΣΗ",
			"ΟΣΗ∆ΗΠΟΤΕ", "ΟΣΗ�?", "ΟΣΗ�?∆ΗΠΟΤΕ", "ΟΣΗΣ", "ΟΣΗΣ∆ΗΠΟΤΕ", "ΟΣΟ", "ΟΣΟ∆ΗΠΟΤΕ", "ΟΣΟΙ", "ΟΣΟΙ∆ΗΠΟΤΕ", "ΟΣΟ�?",
			"ΟΣΟ�?∆ΗΠΟΤΕ", "ΟΣΟΣ", "ΟΣΟΣ∆ΗΠΟΤΕ", "ΟΣΟΥ", "ΟΣΟΥ∆ΗΠΟΤΕ", "ΟΣΟΥΣ", "ΟΣΟΥΣ∆ΗΠΟΤΕ", "ΟΣΩ�?", "ΟΣΩ�?∆ΗΠΟΤΕ",
			"ΟΤΑ�?", "ΟΤΙ", "ΟΤΙ∆ΗΠΟΤΕ", "ΟΤΟΥ", "ΟΥ", "ΟΥ∆Ε", "ΟΥΤΕ", "ΟΧΙ", "ΠΑΛΙ", "ΠΑ�?ΤΟΤΕ", "ΠΑ�?ΤΟΥ", "ΠΑ�?ΤΩΣ",
			"ΠΑΡΑ", "ΠΕΡΑ", "ΠΕΡΙ", "ΠΕΡΙΠΟΥ", "ΠΕΡΙΣΣΟΤΕΡΟ", "ΠΕΡΣΙ", "ΠΕΡΥΣΙ", "ΠΙΑ", "ΠΙΘΑ�?Ο�?", "ΠΙΟ", "ΠΙΣΩ",
			"ΠΛΑΙ", "ΠΛΕΟ�?", "ΠΛΗ�?", "ΠΟΙΑ", "ΠΟΙΑ�?", "ΠΟΙΑΣ", "ΠΟΙΕΣ", "ΠΟΙΟ", "ΠΟΙΟΙ", "ΠΟΙΟ�?", "ΠΟΙΟΣ", "ΠΟΙΟΥ",
			"ΠΟΙΟΥΣ", "ΠΟΙΩ�?", "ΠΟΛΥ", "ΠΟΣΕΣ", "ΠΟΣΗ", "ΠΟΣΗ�?", "ΠΟΣΗΣ", "ΠΟΣΟΙ", "ΠΟΣΟΣ", "ΠΟΣΟΥΣ", "ΠΟΤΕ", "ΠΟΥ",
			"ΠΟΥΘΕ", "ΠΟΥΘΕ�?Α", "ΠΡΕΠΕΙ", "ΠΡΙ�?", "ΠΡΟ", "ΠΡΟΚΕΙΜΕ�?ΟΥ", "ΠΡΟΚΕΙΤΑΙ", "ΠΡΟΠΕΡΣΙ", "ΠΡΟΣ", "ΠΡΟΤΟΥ",
			"ΠΡΟΧΘΕΣ", "ΠΡΟΧΤΕΣ", "ΠΡΩΤΥΤΕΡΑ", "ΠΩΣ", "ΣΑ�?", "ΣΑΣ", "ΣΕ", "ΣΕΙΣ", "ΣΗΜΕΡΑ", "ΣΙΓΑ", "ΣΟΥ", "ΣΤΑ", "ΣΤΗ",
			"ΣΤΗ�?", "ΣΤΗΣ", "ΣΤΙΣ", "ΣΤΟ", "ΣΤΟ�?", "ΣΤΟΥ", "ΣΤΟΥΣ", "ΣΤΩ�?", "ΣΥΓΧΡΟ�?ΩΣ", "ΣΥ�?", "ΣΥ�?ΑΜΑ", "ΣΥ�?ΕΠΩΣ",
			"ΣΥ�?ΗΘΩΣ", "ΣΥΧ�?Α", "ΣΥΧ�?ΑΣ", "ΣΥΧ�?ΕΣ", "ΣΥΧ�?Η", "ΣΥΧ�?Η�?", "ΣΥΧ�?ΗΣ", "ΣΥΧ�?Ο", "ΣΥΧ�?ΟΙ", "ΣΥΧ�?Ο�?", "ΣΥΧ�?ΟΣ",
			"ΣΥΧ�?ΟΥ", "ΣΥΧ�?ΟΥ", "ΣΥΧ�?ΟΥΣ", "ΣΥΧ�?Ω�?", "ΣΥΧ�?ΩΣ", "ΣΧΕ∆Ο�?", "ΣΩΣΤΑ", "ΤΑ", "ΤΑ∆Ε", "ΤΑΥΤΑ", "ΤΑΥΤΕΣ",
			"ΤΑΥΤΗ", "ΤΑΥΤΗ�?", "ΤΑΥΤΗΣ", "ΤΑΥΤΟ,ΤΑΥΤΟ�?", "ΤΑΥΤΟΣ", "ΤΑΥΤΟΥ", "ΤΑΥΤΩ�?", "ΤΑΧΑ", "ΤΑΧΑΤΕ", "ΤΕΛΙΚΑ",
			"ΤΕΛΙΚΩΣ", "ΤΕΣ", "ΤΕΤΟΙΑ", "ΤΕΤΟΙΑ�?", "ΤΕΤΟΙΑΣ", "ΤΕΤΟΙΕΣ", "ΤΕΤΟΙΟ", "ΤΕΤΟΙΟΙ", "ΤΕΤΟΙΟ�?", "ΤΕΤΟΙΟΣ",
			"ΤΕΤΟΙΟΥ", "ΤΕΤΟΙΟΥΣ", "ΤΕΤΟΙΩ�?", "ΤΗ", "ΤΗ�?", "ΤΗΣ", "ΤΙ", "ΤΙΠΟΤΑ", "ΤΙΠΟΤΕ", "ΤΙΣ", "ΤΟ", "ΤΟΙ", "ΤΟ�?",
			"ΤΟΣ", "ΤΟΣΑ", "ΤΟΣΕΣ", "ΤΟΣΗ", "ΤΟΣΗ�?", "ΤΟΣΗΣ", "ΤΟΣΟ", "ΤΟΣΟΙ", "ΤΟΣΟ�?", "ΤΟΣΟΣ", "ΤΟΣΟΥ", "ΤΟΣΟΥΣ",
			"ΤΟΣΩ�?", "ΤΟΤΕ", "ΤΟΥ", "ΤΟΥΛΑΧΙΣΤΟ", "ΤΟΥΛΑΧΙΣΤΟ�?", "ΤΟΥΣ", "ΤΟΥΤΑ", "ΤΟΥΤΕΣ", "ΤΟΥΤΗ", "ΤΟΥΤΗ�?", "ΤΟΥΤΗΣ",
			"ΤΟΥΤΟ", "ΤΟΥΤΟΙ", "ΤΟΥΤΟΙΣ", "ΤΟΥΤΟ�?", "ΤΟΥΤΟΣ", "ΤΟΥΤΟΥ", "ΤΟΥΤΟΥΣ", "ΤΟΥΤΩ�?", "ΤΥΧΟ�?", "ΤΩ�?", "ΤΩΡΑ",
			"ΥΠ", "ΥΠΕΡ", "ΥΠΟ", "ΥΠΟΨΗ", "ΥΠΟΨΙ�?", "ΥΣΤΕΡΑ", "ΦΕΤΟΣ", "ΧΑΜΗΛΑ", "ΧΘΕΣ", "ΧΤΕΣ", "ΧΩΡΙΣ", "ΧΩΡΙΣΤΑ",
			"ΨΗΛΑ", "Ω", "ΩΡΑΙΑ", "ΩΣ", "ΩΣΑ�?", "ΩΣΟΤΟΥ", "ΩΣΠΟΥ", "ΩΣΤΕ", "ΩΣΤΟΣΟ", "ΩΧ"));

	public static final HashSet<String> stopWordsDu = new HashSet<>(Arrays.asList("de", "en", "van", "ik", "te", "dat",
			"die", "in", "een", "hij", "het", "niet", "zijn", "is", "was", "op", "aan", "met", "als", "voor", "had",
			"er", "maar", "om", "hem", "dan", "zou", "of", "wat", "mijn", "men", "dit", "zo", "door", "over", "ze",
			"zich", "bij", "ook", "tot", "je", "mij", "uit", "der", "daar", "haar", "naar", "heb", "hoe", "heeft",
			"hebben", "deze", "u", "want", "nog", "zal", "me", "zij", "nu", "ge", "geen", "omdat", "iets", "worden",
			"toch", "al", "waren", "veel", "meer", "doen", "toen", "moet", "ben", "zonder", "kan", "hun", "dus",
			"alles", "onder", "ja", "eens", "hier", "wie", "werd", "altijd", "doch", "wordt", "wezen", "kunnen", "ons",
			"zelf", "tegen", "na", "reeds", "wil", "kon", "niets", "uw", "iemand", "geweest", "andere"));

	public static final HashSet<String> stopWordsEn = new HashSet<>(Arrays.asList("i", "me", "my", "myself", "we",
			"our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
			"she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
			"what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
			"been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and",
			"but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
			"between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
			"in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
			"where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
			"nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
			"should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
			"hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won",
			"wouldn"));

	public static final HashSet<String> stopWordsSp = new HashSet<>(Arrays.asList("de", "la", "que", "el", "en", "y",
			"a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más",
			"pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre",
			"también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les",
			"ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué",
			"unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos",
			"cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti",
			"tu", "tus", "ellas", "nosotras", "vosostros", "vosostras", "os", "mío", "mía", "míos", "mías", "tuyo",
			"tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras",
			"vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis",
			"están", "esté", "estés", "estemos", "estéis", "estén", "estaré", "estarás", "estará", "estaremos",
			"estaréis", "estarán", "estaría", "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas",
			"estábamos", "estabais", "estaban", "estuve", "estuviste", "estuvo", "estuvimos", "estuvisteis",
			"estuvieron", "estuviera", "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese",
			"estuvieses", "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados",
			"estadas", "estad", "he", "has", "ha", "hemos", "habéis", "han", "haya", "hayas", "hayamos", "hayáis",
			"hayan", "habré", "habrás", "habrá", "habremos", "habréis", "habrán", "habría", "habrías", "habríamos",
			"habríais", "habrían", "había", "habías", "habíamos", "habíais", "habían", "hube", "hubiste", "hubo",
			"hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubiéramos", "hubierais", "hubieran", "hubiese",
			"hubieses", "hubiésemos", "hubieseis", "hubiesen", "habiendo", "habido", "habida", "habidos", "habidas",
			"soy", "eres", "es", "somos", "sois", "son", "sea", "seas", "seamos", "seáis", "sean", "seré", "serás",
			"será", "seremos", "seréis", "serán", "sería", "serías", "seríamos", "seríais", "serían", "era", "eras",
			"éramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos", "fuisteis", "fueron", "fuera", "fueras",
			"fuéramos", "fuerais", "fueran", "fuese", "fueses", "fuésemos", "fueseis", "fuesen", "sintiendo", "sentido",
			"sentida", "sentidos", "sentidas", "siente", "sentid", "tengo", "tienes", "tiene", "tenemos", "tenéis",
			"tienen", "tenga", "tengas", "tengamos", "tengáis", "tengan", "tendré", "tendrás", "tendrá", "tendremos",
			"tendréis", "tendrán", "tendría", "tendrías", "tendríamos", "tendríais", "tendrían", "tenía", "tenías",
			"teníamos", "teníais", "tenían", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera",
			"tuvieras", "tuviéramos", "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuviésemos", "tuvieseis",
			"tuviesen", "teniendo", "tenido", "tenida", "tenidos", "tenidas", "tened"));

}
