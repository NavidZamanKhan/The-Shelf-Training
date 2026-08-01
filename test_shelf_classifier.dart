import 'dart:convert';
import 'dart:io';
import 'dart:math';

class ShelfClassifier {
  final List<String> classes;
  final Map<String, int> vocabulary;
  final List<double> idf;
  final List<List<double>> coef;
  final List<double> intercept;
  final bool sublinearTf;

  ShelfClassifier({
    required this.classes,
    required this.vocabulary,
    required this.idf,
    required this.coef,
    required this.intercept,
    this.sublinearTf = true,
  });

  factory ShelfClassifier.fromJson(Map<String, dynamic> json) {
    final classesList = List<String>.from(json['classes']);
    final vocabMap = Map<String, int>.from(json['vocabulary']);
    final idfList = (json['idf'] as List).map((e) => (e as num).toDouble()).toList();
    final coefList = (json['coef'] as List)
        .map((row) => (row as List).map((e) => (e as num).toDouble()).toList())
        .toList();
    final interceptList =
        (json['intercept'] as List).map((e) => (e as num).toDouble()).toList();
    final sublinear = json['sublinear_tf'] ?? true;

    return ShelfClassifier(
      classes: classesList,
      vocabulary: vocabMap,
      idf: idfList,
      coef: coefList,
      intercept: interceptList,
      sublinearTf: sublinear,
    );
  }

  /// Tokenizes string into lowercased alphanumeric words and extracts 1-grams and 2-grams.
  List<String> _extractTokens(String text) {
    // Regex matching word tokens across English & Unicode (Bangla, etc.)
    final RegExp wordRegExp = RegExp(r'[\p{L}\p{N}]+', unicode: true);
    final Iterable<Match> matches = wordRegExp.allMatches(text.toLowerCase());
    final List<String> words = matches.map((m) => m.group(0)!).toList();

    final List<String> tokens = [];
    for (int i = 0; i < words.length; i++) {
      tokens.add(words[i]);
      if (i < words.length - 1) {
        tokens.add('${words[i]} ${words[i + 1]}');
      }
    }
    return tokens;
  }

  /// Classifies text input and returns the predicted category label & probability distribution.
  Map<String, dynamic> predict(String text) {
    final List<String> tokens = _extractTokens(text);

    // Count term frequencies for terms matching vocabulary
    final Map<int, int> termCounts = {};
    for (final token in tokens) {
      if (vocabulary.containsKey(token)) {
        final int idx = vocabulary[token]!;
        termCounts[idx] = (termCounts[idx] ?? 0) + 1;
      }
    }

    // Build sublinear TF * IDF sparse vector
    final Map<int, double> sparseVec = {};
    double sumSq = 0.0;

    termCounts.forEach((idx, count) {
      final double tf = sublinearTf ? (1.0 + log(count)) : count.toDouble();
      final double val = tf * idf[idx];
      sparseVec[idx] = val;
      sumSq += val * val;
    });

    // L2 Normalize
    final double l2Norm = sqrt(sumSq);
    if (l2Norm > 0) {
      sparseVec.updateAll((key, val) => val / l2Norm);
    }

    // Dot product: scores = (coef * sparseVec) + intercept
    final int numClasses = classes.length;
    final List<double> scores = List<double>.from(intercept);

    sparseVec.forEach((featureIdx, featureVal) {
      for (int c = 0; c < numClasses; c++) {
        scores[c] += coef[c][featureIdx] * featureVal;
      }
    });

    // Softmax to compute probabilities
    double maxScore = scores.reduce(max);
    double expSum = 0.0;
    final List<double> expScores = List.filled(numClasses, 0.0);

    for (int c = 0; c < numClasses; c++) {
      final double expVal = exp(scores[c] - maxScore);
      expScores[c] = expVal;
      expSum += expVal;
    }

    final Map<String, double> probabilities = {};
    int bestClassIdx = 0;
    double bestProb = 0.0;

    for (int c = 0; c < numClasses; c++) {
      final double prob = expScores[c] / expSum;
      probabilities[classes[c]] = prob;
      if (prob > bestProb) {
        bestProb = prob;
        bestClassIdx = c;
      }
    }

    return {
      'predicted_label': classes[bestClassIdx],
      'confidence': bestProb,
      'probabilities': probabilities,
    };
  }
}

void main() async {
  print('=== DART ON-DEVICE SHELF CLASSIFIER PROOF-OF-CONCEPT ===\n');

  final file = File('output/tfidf_model.json');
  if (!file.existsSync()) {
    print('Error: output/tfidf_model.json not found. Run export_model.py first.');
    exit(1);
  }

  print('Loading model parameters from output/tfidf_model.json...');
  final stopwatch = Stopwatch()..start();
  final String jsonStr = await file.readAsString();
  final Map<String, dynamic> jsonMap = jsonDecode(jsonStr);
  final classifier = ShelfClassifier.fromJson(jsonMap);
  stopwatch.stop();

  print('Model loaded in ${stopwatch.elapsedMilliseconds} ms.');
  print('Classes count: ${classifier.classes.length}');
  print('Vocabulary size: ${classifier.vocabulary.length} terms\n');

  final List<String> testPrompts = [
    'A dark wizard threatens the magical kingdom with ancient dark spells',
    'The history of World War II and European political alliances in the 20th century',
    'A detective investigates a mysterious murder mystery in a foggy city',
    'A collection of romantic poems expressing deep love, heartbreak, and emotional devotion',
    'Deep philosophical thoughts on human consciousness, morality, and existence',
    'একটি ভয়ংকর পুরনো রাজবাড়িতে অশরীরী ভূতের রহস্যময় উপদ্রব ও আতঙ্কের ঘটনা...',
    'দুটি তরুণ হৃদয়ের গভীর ভালোবাসা, প্রেম, ব্যাকুলতা ও আবেগঘন বিরহের কাহিনী...',
    'ভবিষ্যতের মহাকাশ অভিযান, ভিনগ্রহের প্রাণী ও রোবটের বৈজ্ঞানিক কল্পকাহিনী...',
  ];

  print('------------------------------------------------------------');
  print('DART INFERENCE EVALUATION:');
  print('------------------------------------------------------------');

  for (int i = 0; i < testPrompts.length; i++) {
    final prompt = testPrompts[i];
    final sw = Stopwatch()..start();
    final result = classifier.predict(prompt);
    sw.stop();

    final displayPrompt = prompt.length > 60 ? prompt.substring(0, 60) + "..." : prompt;
    final label = result['predicted_label'];
    final conf = (result['confidence'] * 100).toStringAsFixed(2);
    final ms = (sw.elapsedMicroseconds / 1000.0).toStringAsFixed(3);

    print('[Input ${i + 1}]: "$displayPrompt"');
    print('  -> Predicted Shelf: [$label] (Confidence: $conf%)');
    print('  -> Inference Time: $ms ms\n');
  }

  print('------------------------------------------------------------');
  print('Dart verification test complete successfully!');
}
