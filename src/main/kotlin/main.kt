import BINARIZATION.binaryThreshold
import CONTRAST.alpha
import CONTRAST.beta
import CONTRAST.rType
import CONTRAST2.alpha2
import CONTRAST2.beta2
import DICT.alignedYMargin
import DICT.dictionaryProperties
import DICT.dictionaryX
import DICT.dictionaryY
import DICT.distanceThresh
import DICT.levenhtein
import DICT.lineMergingYDistance
import DICT.lineMergingYDistanceForValuesAndPossibleMUOnNextLine
import DICT.muSet
import DICT.numberOfRowsToAddToTheActualNumberOfRows
import DICT.shrinkedList
import DICT.words
import DictionaryType.X
import DictionaryType.Y
import FileUtils.getImg
import HOUGH.angleResolutionInRadians
import HOUGH.distanceResolutionInPixels
import HOUGH.finalTheta
import HOUGH.houghCounter
import HOUGH.lines
import HOUGH.minimumVotes
import HOUGH.minimumVotesStep
import HOUGH.minimumVotesToConsider
import HOUGH.scalar
import IMG.originalImg
import IMG.originalImgNotResized
import IMG.properties
import IMG.resizeRatio
import LINESHRINKING.maxRhoThereshold
import info.debatty.java.stringsimilarity.NormalizedLevenshtein
import net.sourceforge.tess4j.Word
import org.bytedeco.javacpp.indexer.FloatRawIndexer
import org.bytedeco.javacpp.indexer.UByteIndexer
import org.bytedeco.javacpp.opencv_core.*
import org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED
import org.bytedeco.javacpp.opencv_imgcodecs.imread
import org.bytedeco.javacpp.opencv_imgproc.*
import org.bytedeco.javacv.OpenCVFrameConverter
import java.awt.Rectangle
import java.lang.Math.*


object DICT {
    val dictionaryProperties = listOf("energia", "energetico", "grassi", "acidi", "saturi", "insaturi", "monoinsaturi", "polinsaturi", "carboidrati", "zuccheri", "proteine", "fibre", "sale", "sodio", "fibre", "fibra", "alimentare", "amido")
    val dictionaryY = listOf("informazioni", "tabella", "dichiarazione", "nutrizionale", "nutrizionali")
    val dictionaryX = listOf("energia", "grassi", "carboidrati", "proteine", "sale")
    val words = mutableListOf<Word>()
    val levenhtein = NormalizedLevenshtein()
    val shrinkedList = mutableListOf<Word>()
    const val distanceThresh = 0.5
    const val lineMergingYDistance = 25
    const val lineMergingYDistanceForValuesAndPossibleMUOnNextLine = 15
    const val numberOfRowsToAddToTheActualNumberOfRows = 20
    const val alignedYMargin = 50
    val muSet = setOf("8", "9", "g", "kcal", "kJ")
}

enum class DictionaryType {
    X, Y
}

enum class BinarizationType {
    OTSU, BINARY
}

object IMG {
    private const val imgName = "colore" + "/" + "training" + "/" + "panna" + "." + "jpg"
    const val resizeRatio = 0.5
    val originalImgNotResized: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED)
    val originalImg: Mat = imread(getImg(imgName), CV_LOAD_IMAGE_UNCHANGED).also { it.resizeSelf() }
    val imgConverter = OpenCVFrameConverter.ToMat()
    var properties = mutableListOf<CustomDistance>()
}

object CONTRAST {
    const val rType = -1      //  desired output matrix type or, rather, the depth since the number of channels are the same as the input has; if rtype is negative, the output matrix will have the same type as the input.
    const val alpha = 1.55 // optional scale factor.
    const val beta = -25.5 // optional delta added to the scaled values.
}

object CONTRAST2 {
    val rType = -1 //  desired output matrix type or, rather, the depth since the number of channels are the same as the input has; if rtype is negative, the output matrix will have the same type as the input.
    val alpha2 = 1.5 // optional scale factor.
    val beta2 = -25.0 // optional delta added to the scaled values.
}

object BINARIZATION {
    const val binaryThreshold = 110.0
}

object HOUGH {
    var houghCounter = 1
    const val distanceResolutionInPixels: Double = 1.0 // rho
    const val angleResolutionInRadians: Double = PI / 180 // theta
    var minimumVotes = 1000
    var minimumVotesToConsider = 3
    var minimumVotesStep = 100
    var finalTheta: Double = 0.0 // degrees
    var lines = Mat()
    val scalar = Scalar(0.0, 0.0, 255.0, 0.0)
}

object LINESHRINKING {
    const val maxRhoThereshold = 500
}

fun main(args: Array<String>) {
    runMainAlgorithm()
}

fun runMainAlgorithm() {
    applyHoughWithPreprocessing(originalImgNotResized.clone().resizeSelf())

    // correct rotation w/ average horizontal 0
    originalImgNotResized.rotateToTheta()
    originalImg.rotateToTheta()
    originalImg.resizeSelf().show("rotated original")

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NB now originalImg is rotated, let's re-run the previous steps //////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
    val imgForOCR = originalImgNotResized.clone()
    applyPreprocessingForOCR(imgForOCR, BinarizationType.BINARY)
    val rect = getRectForCrop(imgForOCR)
    val croppedOriginalNotResized = crop(originalImgNotResized, rect)
    croppedOriginalNotResized.clone().resizeSelf().resizeSelf().show("cropped")

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NB now original image is cropped to be only the table ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // correct rotation w/ average horizontal 0
    applyHoughWithPreprocessing(croppedOriginalNotResized.clone().resizeSelf())
    croppedOriginalNotResized.rotateToTheta()
    croppedOriginalNotResized.clone().resizeSelf().resizeSelf().show("Final rotation (resized and cropped)")

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NB now original image is rotated and ready for final ocr ///////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    val imgForOCR2 = croppedOriginalNotResized.clone()
    applyPreprocessingForOCR(imgForOCR2, BinarizationType.BINARY)

    words.clear()
    words.addAll(imgForOCR2.getWords())
    extractNutritionalPropertyNames()
    extractNutritionalPropertiesValues()
    mergePropertiesWithValues()
    */
}

fun applyHoughWithPreprocessing(source: Mat, imgToBeUsedForDisplayingLines: Mat = source.clone()) {
    // convert to greyscale
    convertToGreyscale(source)
    source.clone().resizeSelf().show("grayscale preHough")

    increaseContrast(source)
    source.clone().resizeSelf().show("contrast preHough")

    // apply sobel + otsu edge detection
    applySobel(source)
    applyOtsu(source)
    source.resizeSelf().show("sobelX")

    // apply Hough transform
    applyHough(source, imgToBeUsedForDisplayingLines)
}

fun applyPreprocessingForOCR(source: Mat, binType: BinarizationType) {
    // reduce color
    reduceColor(source)
    source.clone().resizeSelf().resizeSelf().show("reduce color forOCR 1")

    // convert to greyscale
    convertToGreyscale(source)
    source.clone().resizeSelf().resizeSelf().show("greyscale forOCR")

    // convert to greyscale
    increaseContrast2(source)
    source.clone().resizeSelf().resizeSelf().show("contrast forOCR")

    // apply otsu binary filter
    if (binType == BinarizationType.OTSU) {
        applyOtsu(source)
        source.clone().resizeSelf().resizeSelf().show("otsu forOCR")
    } else {
        applyBinary(source)
        source.clone().resizeSelf().resizeSelf().show("After binarization (non otsu) forOCR")
    }
}

fun Mat.resizeSelf(): Mat {
    resize(this, this, Size((this.size().width() * resizeRatio).toInt(), (this.size().height() * resizeRatio).toInt()))
    return this
}

fun convertToGreyscale(source: Mat, dest: Mat = source) = cvtColor(source, dest, CV_BGR2GRAY)

fun increaseContrast(source: Mat) = source.convertTo(source, rType, alpha, beta)

fun increaseContrast2(source: Mat) = source.convertTo(source, rType, alpha2, beta2)

fun applySobel(source: Mat) {
    val gradX = Mat()
    val gradY = Mat()
    val absGradX = Mat()
    val absGradY = Mat()
    Sobel(source, gradX, CV_16S, 1, 0, 3, 1.0, 0.0, BORDER_DEFAULT)
    Sobel(source, gradY, CV_16S, 0, 1, 3, 1.0, 0.0, BORDER_DEFAULT)
    convertScaleAbs(gradX, absGradX)
    convertScaleAbs(gradY, absGradY)
    addWeighted(absGradX, 0.5, absGradY, 0.5, 0.0, source)
}

fun reduceColor(source: Mat) {
    val div = 64
    val indexer = source.createIndexer() as UByteIndexer
    // Total number of elements, combining components from each channel
    val nbElements = source.rows() * source.cols() * source.channels()
    for (i in 0 until nbElements) {
        // Convert to integer, byte is treated as an unsigned value
        val v = indexer.get(i.toLong())
        // Use integer division to reduce number of values
        val newV = v / div * div + div / 2
        // Put back into the image
        indexer.put(i.toLong(), newV)
    }
}

fun Mat.rotateToTheta() {
    if (finalTheta != 0.0) {
        warpAffine(this, this, getRotationMatrix2D(Point2f((this.size().width() / 2).toFloat(), (this.size().height() / 2).toFloat()), finalTheta, 1.0), this.size())
    }
}

fun applyOtsu(source: Mat, dest: Mat = source) = threshold(source, dest, 0.0, 255.0, THRESH_BINARY + THRESH_OTSU)

fun applyBinary(source: Mat, dest: Mat = source) = threshold(source, dest, binaryThreshold, 255.0, THRESH_BINARY)

fun crop(source: Mat, rect: Rect): Mat = Mat(source, rect)

fun applyHough(source: Mat, matForDisplay: Mat) {

    lines = Mat()
    HoughLines(source, lines, distanceResolutionInPixels, angleResolutionInRadians, minimumVotes)

    val indexer = lines.createIndexer() as FloatRawIndexer
    val houghResult = Mat().also { source.copyTo(it) }
    val horizontalLinesList = mutableListOf<Line>()
    val verticalLinesList = mutableListOf<Line>()

    for (i in 0 until lines.rows()) {
        val rho = indexer.get(i.toLong(), 0, 0)
        val thetaDeg = indexer.get(i.toLong(), 0, 1).toDouble().toDegrees()
        var theta = indexer.get(i.toLong(), 0, 1).toDouble()
        lateinit var p1: Point
        lateinit var p2: Point

        if (thetaDeg <= 45 || thetaDeg >= 135) {
            // ~vertical line
            if (thetaDeg < 10 || thetaDeg > 170) {
                p1 = Point(round(rho / cos(theta)).toInt(), 0) // point of intersection of the line with first row
                p2 = Point(
                    round((rho - houghResult.rows() * sin(theta)) / cos(theta)).toInt(),
                    houghResult.rows()
                ) // point of intersection of the line with last row
                if (thetaDeg > 90) theta = -(PI - theta)
                verticalLinesList.add(Line(rho, theta, p1, p2))
            }
        } else {
            // ~horizontal line
            if ((thetaDeg < 95 && thetaDeg > 85)) {
                p1 = Point(0, round(rho / sin(theta)).toInt()) // point of intersection of the line with first column
                p2 = Point(
                    houghResult.cols(),
                    round((rho - houghResult.cols() * cos(theta)) / sin(theta)).toInt()
                ) // point of intersection of the line with last column
                horizontalLinesList.add(Line(rho, theta, p1, p2))
            }
        }
    }

    horizontalLinesList.addAll(horizontalLinesList.sortedBy { it.rho }.toMutableList())
    verticalLinesList.addAll(verticalLinesList.sortedBy { it.rho }.toMutableList())

    fun removeLines(list: MutableList<Line>) {
        for (i in 0 until list.size) {
            for (j in i + 1 until list.size - 1) {
                if (j < list.size) {
                    val shouldBeRemoved = Math.abs(list[i].rho - list[j].rho) < maxRhoThereshold
                    if (shouldBeRemoved) {
                        list.removeAt(j)
                    } else break
                }
            }
        }
    }

    removeLines(horizontalLinesList)
    removeLines(verticalLinesList)

    val vertTheta = if (verticalLinesList.isNotEmpty()) verticalLinesList.map { it.theta.toDegrees() }.average() else 0.0
    val horTheta = if (horizontalLinesList.isNotEmpty()) horizontalLinesList.map { it.theta.toDegrees() }.average() - 90 else 0.0
    finalTheta = listOf(vertTheta, horTheta).average()
    println("$vertTheta $horTheta $finalTheta")

    val res2 = matForDisplay.clone()
    //draw lines
    horizontalLinesList.forEach { line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0) }
    verticalLinesList.forEach { line(res2, it.p1, it.p2, scalar, 1, LINE_8, 0) }


    if (verticalLinesList.size < minimumVotesToConsider && horizontalLinesList.size < minimumVotesToConsider) {
        minimumVotes -= minimumVotesStep
        applyHough(source, matForDisplay)
    } else {
        res2.clone().resizeSelf().show("applied hough n°$houghCounter | minimumVotes: $minimumVotes")
        houghCounter += 1
        minimumVotes = 1000
    }
}

fun getRectForCrop(source: Mat): Rect {
    val cropOffset = 30

    words.clear()
    words.addAll(source.getWords())

    fun getWordForCrop(dictionaryType: DictionaryType): CustomDistance {
        val dictionary = when (dictionaryType) {
            X -> dictionaryX
            Y -> dictionaryY
        }

        return words
            .map { word ->
                dictionary
                    .map { dictWord ->
                        val distance = levenhtein.distance(word.text.toLowerCase(), dictWord)
                        var boxCoordinate = if (dictionaryType == X) word.boundingBox.x else word.boundingBox.y
                        boxCoordinate = if (boxCoordinate == 0) 10 else boxCoordinate
                        val weight = distance * boxCoordinate
                        CustomDistance(word, dictWord, levenhtein.distance(word.text.toLowerCase(), dictWord), weight)
                    }
                    .minBy { it.weight }!!
            }
            .filter { it.weight < 0.5 }
            .sortedWith(if (dictionaryType == X) compareBy({ it.weight }, { it.ocrWord.boundingBox.x }) else compareBy({ it.weight }, { it.ocrWord.boundingBox.y }))
            .onEach { println(it) }
            .minWith(if (dictionaryType == X) compareBy({ it.weight }, { it.ocrWord.boundingBox.x }) else compareBy({ it.weight }, { it.ocrWord.boundingBox.y }))
                ?: CustomDistance(Word("", 0.0.toFloat(), Rectangle(0, 0, 0, 0)), "", 0.0)
    }

    //Computes the levenstein distance for each word with selected dictionaries which contain the keywords for cropping
    val wordY = getWordForCrop(Y)
    printlndiv()
    printlndiv()
    printlndiv()
    val wordX = getWordForCrop(X)
    var x = wordX.ocrWord.boundingBox.x - cropOffset
    var y = wordY.ocrWord.boundingBox.y - cropOffset
    if (x < 0) x = 0
    if (y < 0) y = 0

//    words.sortedBy { it.boundingBox.x }.forEach { println(it) }

    println("words used for crop: | $wordY | \nand\n | $wordX |")

    return Rect(x, y, source.size().width() - x, source.size().height() - y)
}

fun extractNutritionalPropertyNames() {

    words.forEach { println(it.text + " y:" + it.boundingBox.y + " --conf: " + levenhtein.distance(it.text.toLowerCase(), "grassi")) }
    //Foreach word found by tesseract, it saves in a different structure only the words that are equal or similar to the ones in the dictionary
    words
        .forEach { ocrWord ->
            dictionaryProperties
                .map { dictWord -> CustomDistance(ocrWord, dictWord, levenhtein.distance(ocrWord.text.toLowerCase(), dictWord)) }
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

    printlndiv()
    properties.forEach { println(it) }

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
                            newProperties.last().ocrWord.boundingBox.height + properties[j].ocrWord.boundingBox.height
                        )
                    ),
                    newProperties.last().dictWord + " " + properties[j].dictWord,
                    min(newProperties.last().distance, properties[j].distance)
                )
                alreadyMergedPropertiesIndexes.add(j)
                j++
            }
        }
    }

    val averageMostAccurateX = newProperties.filter { it.distance < 0.3 }.map { it.ocrWord.boundingBox.x }.average()
    //properties.forEach { println(it) }
    properties = newProperties.filter { Math.abs(it.ocrWord.boundingBox.x - averageMostAccurateX) < 250 }.toMutableList()
    println(".......................:")
    properties.forEach { println(it) }
}

fun extractNutritionalPropertiesValues() {

    val maxXX = properties.maxBy { it.ocrWord.boundingBox.x + it.ocrWord.boundingBox.width }!!
    println("LA WORD MAx X " + maxXX.dictWord + "--//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////---")
    //X value from which starting to look for nutritional values
    val maxX = maxXX.ocrWord.boundingBox.x + maxXX.ocrWord.boundingBox.width + 150
    // probably there's no need for this one because properties are already shrinked in lines before
    //val numRows = properties.mapIndexed { i, p -> if (i + 1 < properties.size) Math.abs(p.ocrWord.boundingBox.y - properties[i + 1].ocrWord.boundingBox.y) else 0 }.filter { it > 5 }.count() + 1
    val numRows = properties.size

    // Selects only the Words that have the smallest X values (as many as the number of proprieties (plus a margin of error)
    var allRightValues = words.filter { it.text.contains(Regex("[0-9]")) }.filter { it.boundingBox.x > maxX }.sortedBy { it.boundingBox.x }.sortedBy { it.boundingBox.y }.take(numRows + numberOfRowsToAddToTheActualNumberOfRows)
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

    printlndiv()
    shrinkedList.sortedBy { it.boundingBox.x }.take(numRows + 5).sortedBy { it.boundingBox.y }.forEach { println(it) }
    printlndiv()
}

fun mergePropertiesWithValues() {

    val map = mutableMapOf<CustomDistance, Word>()

    //For each property looks for a corresponding value almost on the same Y value (aligned)
    properties.forEach { prop ->
        shrinkedList
            .forEach { value ->
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

}

data class Line(val rho: Float, val theta: Double, val p1: Point, val p2: Point)

data class CustomDistance(val ocrWord: Word, val dictWord: String, val distance: Double, val weight: Double = 10.0)