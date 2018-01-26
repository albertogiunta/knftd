import info.debatty.java.stringsimilarity.NormalizedLevenshtein
import net.sourceforge.tess4j.Word
import org.bytedeco.javacpp.opencv_core
import java.awt.Rectangle

enum class DictionaryType {
    X, Y
}

data class Line(val rho: Float, val theta: Double, val p1: opencv_core.Point, val p2: opencv_core.Point)

data class CustomDistance(val ocrWord: Word, val dictWord: String, val distance: Double, val weight: Double = 10.0)

class OCRProcessor {

    private val dictionaryProperties = listOf("energia", "energetico", "grassi", "acidi", "saturi", "insaturi", "monoinsaturi", "polinsaturi", "carboidrati", "zuccheri", "proteine", "fibre", "sale", "fibre", "fibra", "alimentare")
    private val dictionaryY = listOf("informazioni", "tabella", "dichiarazione", "nutrizionale", "nutrizionali")
    private val dictionaryX = listOf("energia", "grassi", "carboidrati", "proteine", "sale")
    private val distanceThresh = 0.5
    private val lineMergingYDistance = 50
    private val lineMergingYDistanceForValuesAndPossibleMUOnNextLine = 15
    private val numberOfRowsToAddToTheActualNumberOfRows = 20
    private val alignedYMargin = 30
    private val levenshtein = NormalizedLevenshtein()
    private val muSet = setOf("8", "9", "g", "kcal", "kJ")


    fun extractNutritionalPropertyNames(words : List<Word>) : List<CustomDistance> {
        var properties = mutableListOf<CustomDistance>()
        //words.forEach { println(it.text + " y:" + it.boundingBox.y + " --conf: " + levenshtein.distance(it.text.toLowerCase(), "grassi")) }
        //Foreach word found by tesseract, it saves in a different structure only the words that are equal or similar to the ones in the dictionary
        words
                .forEach { ocrWord ->
                    dictionaryProperties
                            .map { dictWord -> CustomDistance(ocrWord, dictWord, levenshtein.distance(ocrWord.text.toLowerCase(), dictWord)) }
                            .filter { it.distance <= distanceThresh }
                            .forEach { properties.add(it) }
                }

        //Contains indexes of words that have already been merged with a previous word
        val alreadyMergedPropertiesIndexes = mutableListOf<Int>()
        //Contains merged proprieties (like "acidi grassi saturi")
        val newProperties = mutableListOf<CustomDistance>()
        //Filters the words that match with more than one dictionary word, keeping only the one that matches more (has the smallest distance)
        val propertiesToKeep = properties
                .map { main ->
                    properties
                            .filter { it.ocrWord == main.ocrWord }
                            .minBy { it.distance }!!
                }.sortedBy { it.distance }
                .distinctBy { it.dictWord }
                .toMutableList()
        val propToRemove = properties.subtract(propertiesToKeep)
        properties = properties.subtract(propToRemove.filterNot { it.dictWord == "grassi" }).toMutableList()

        //printlndiv()
        //properties.forEach { println(it) }

        properties.forEachIndexed { i, _ ->
            //If the word has not been merged already
            if (!alreadyMergedPropertiesIndexes.contains(i)) {
                //Save the Word representing the property
                newProperties.add(properties[i])
                var j = i + 1
                //Check next Words, if they have almost the same Y values it means that they are on the same line, and it add the text to the current Word text
                while (j < properties.size - 1 && Math.abs(properties[i].ocrWord.boundingBox.y - properties[j].ocrWord.boundingBox.y) < lineMergingYDistance) {
                    newProperties[newProperties.size - 1] = CustomDistance(
                            Word(
                                    properties[j].ocrWord.text,
                                    properties[j].ocrWord.confidence,
                                    Rectangle(
                                            newProperties.last().ocrWord.boundingBox.x,
                                            newProperties.last().ocrWord.boundingBox.y,
                                            newProperties.last().ocrWord.boundingBox.width + properties[j].ocrWord.boundingBox.width,
                                            newProperties.last().ocrWord.boundingBox.height
                                    )
                            ),
                            newProperties.last().dictWord + " " + properties[j].dictWord,
                            Math.min(newProperties.last().distance, properties[j].distance)
                    )
                    alreadyMergedPropertiesIndexes.add(j)
                    j++
                }
            }
        }

        val averageMostAccurateX = newProperties.filter { it.distance < 0.5 }.map { it.ocrWord.boundingBox.x }.average()
        //properties.forEach { println(it) }
        properties = newProperties.filter { Math.abs(it.ocrWord.boundingBox.x - averageMostAccurateX) < 250 }.toMutableList()
        println(".......................:")
        properties.forEach { println(it) }

        return properties.toList()
    }

    fun extractNutritionalPropertiesValues(words : List<Word>,properties : List<CustomDistance>) : List<Word>{
        val shrinkedList = mutableListOf<Word>()
        val maxXX = properties.maxBy { it.ocrWord.boundingBox.x + it.ocrWord.boundingBox.width }!!
        println("LA WORD MAx X " + maxXX.dictWord + "--//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////---")
        //X value from which starting to look for nutritional values
        val maxX = maxXX.ocrWord.boundingBox.x + maxXX.ocrWord.boundingBox.width
        // probably there's no need for this one because properties are already shrinked in lines before
        //val numRows = properties.mapIndexed { i, p -> if (i + 1 < properties.size) Math.abs(p.ocrWord.boundingBox.y - properties[i + 1].ocrWord.boundingBox.y) else 0 }.filter { it > 5 }.count() + 1
        val numRows = properties.size

        // Selects only the Words that have the smallest X values (as many as the number of proprieties (plus a margin of error)
        var allRightValues = words.filter { it.text.contains(Regex("[0-9]")) }.filter { it.boundingBox.x > maxX }//.sortedBy { it.boundingBox.x }.sortedBy { it.boundingBox.y }//.take(numRows + numberOfRowsToAddToTheActualNumberOfRows)
        //allRightValues = allRightValues.filterNot { it.text.contains(Regex("A-Za-z0-9")) }
        allRightValues = allRightValues.filterNot { it.text.contains(Regex("[/, .]")) && !it.text.contains(Regex("[A-Za-z0-9]")) }
        //Fixes the error made with measurement units (often alone in one line, often a "9", etc) checking for each Word if the next one is a 9 and merging
        allRightValues.mapIndexed { i, p ->
            if (i + 1 < allRightValues.size) {
                if ((muSet.contains(allRightValues[i + 1].text)) && Math.abs(p.boundingBox.y - allRightValues[i + 1].boundingBox.y) < lineMergingYDistanceForValuesAndPossibleMUOnNextLine) {
                    shrinkedList.add(Word(p.text + " g", p.confidence, p.boundingBox))
                } else if (!muSet.contains(p.text)) {
                    shrinkedList.add(Word(p.text, p.confidence, p.boundingBox))
                }
            }
        }

        //printlndiv()
        //shrinkedList.forEach { println(it) }
        //printlndiv()

        return shrinkedList.toList()
    }

    fun mergePropertiesWithValues(properties : List<CustomDistance>, shrinkedList : List<Word>) : Map<CustomDistance, Word>{

        val map = mutableMapOf<CustomDistance, Word>()

        //For each property looks for a corresponding value almost on the same Y value (aligned)
        properties.forEach { prop ->
            shrinkedList
                    .forEach { value ->
                        //if(Math.abs(prop.ocrWord.boundingBox.y - value.boundingBox.y) < 101)
                        println("${prop.dictWord} - ${value.text} |  prop y ${prop.ocrWord.boundingBox.y} | value y ${value.boundingBox.y}")
                        if (Math.abs(value.boundingBox.y - prop.ocrWord.boundingBox.y) < alignedYMargin && (!map.containsKey(prop) || value.boundingBox.x < map[prop]!!.boundingBox.x)) {
                            map[prop] = value
                        }
                    }
        }

        map.forEach { i, u ->
            if (u.text.endsWith(",9") || (!u.text.endsWith("g") && !u.text.endsWith("kcal") && !u.text.endsWith("kJ")) && !u.text.endsWith("9")) {
                map[i] = Word(u.text + " g", u.confidence, u.boundingBox)
            } else if (u.text.endsWith("9") || u.text.endsWith("Q")) {
                map[i] = Word(u.text.dropLast(1) + " g", u.confidence, u.boundingBox)
            }
        }

        printlndiv()
        map.forEach { t, u -> println(t.dictWord + " " + u.text) }

        return map.toMap()

    }

    fun extractWords(source: opencv_core.Mat) : List<Word> {
        return source.getWords()
    }


    fun getRectForCrop(source: opencv_core.Mat, words : List<Word>): opencv_core.Rect {
        val cropOffset = 30

        words.toMutableList().clear()
        words.toMutableList().addAll(source.getWords())

        fun getWordForCrop(dictionaryType: DictionaryType): CustomDistance {

            val dictionary = when (dictionaryType) {
                DictionaryType.X -> dictionaryX
                DictionaryType.Y -> dictionaryY
            }

            return words
                    .map { word ->
                        dictionary
                                .map { dictWord ->
                                    val distance = levenshtein.distance(word.text.toLowerCase(), dictWord)
                                    var boxCoordinate = if (dictionaryType == DictionaryType.X) word.boundingBox.x else word.boundingBox.y
                                    boxCoordinate = if (boxCoordinate == 0) 10 else boxCoordinate
                                    val weight = distance * boxCoordinate
                                    CustomDistance(word, dictWord, levenshtein.distance(word.text.toLowerCase(), dictWord), weight)
                                }
                                .minBy { it.weight }!!
                    }
                    .filter { it.weight < 0.5 }
                    .sortedWith(if (dictionaryType == DictionaryType.X) compareBy({ it.weight }, { it.ocrWord.boundingBox.x }) else compareBy({ it.weight }, { it.ocrWord.boundingBox.y }))
                    //.onEach { println(it) }
                    .minWith(if (dictionaryType == DictionaryType.X) compareBy({ it.weight }, { it.ocrWord.boundingBox.x }) else compareBy({ it.weight }, { it.ocrWord.boundingBox.y }))
                    ?: CustomDistance(Word("", 0.0.toFloat(), Rectangle(0, 0, 0, 0)), "", 0.0)
        }

        //Computes the levenstein distance for each word with selected dictionaries which contain the keywords for cropping
        val wordY = getWordForCrop(DictionaryType.Y)
        val wordX = getWordForCrop(DictionaryType.X)
        var x = wordX.ocrWord.boundingBox.x - cropOffset
        var y = wordY.ocrWord.boundingBox.y - cropOffset
        if (x < 0) x = 0
        if (y < 0) y = 0

//    words.sortedBy { it.boundingBox.x }.forEach { println(it) }

        println("words used for crop: | $wordY | \nand\n | $wordX |")

        return opencv_core.Rect(x, y, source.size().width() - x, source.size().height() - y)
    }

}
