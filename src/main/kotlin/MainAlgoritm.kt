import FileUtils.getImg
import net.sourceforge.tess4j.Word
import org.bytedeco.javacpp.opencv_core
import org.bytedeco.javacpp.opencv_imgcodecs
import org.bytedeco.javacpp.opencv_imgproc

class MainAlgorithm(imgName : String){

    private val originalImgNotResized: opencv_core.Mat = opencv_imgcodecs.imread(getImg(imgName), opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED)
    private val originalImg: opencv_core.Mat = opencv_imgcodecs.imread(getImg(imgName), opencv_imgcodecs.CV_LOAD_IMAGE_UNCHANGED).also { it.resizeSelf() }
    private val imageProcessor = ImageProcessor()
    private val ocrProcessor = OCRProcessor()
    private val words = mutableListOf<Word>()

    fun run(){

        originalImg.show("Original")
        applyHoughWithPreprocessing(originalImgNotResized.clone().resizeSelf())

        rotateImage()

        val imgForOCR = originalImgNotResized.clone()
        applyPreprocessingForOCR(imgForOCR, BinarizationType.BINARY)
        words.addAll(ocrProcessor.extractWords(imgForOCR))

        val rect = ocrProcessor.getRectForCrop(imgForOCR, words)
        val croppedOriginalNotResized = imageProcessor.crop(originalImgNotResized, rect)
        //croppedOriginalNotResized.clone().resizeSelf().resizeSelf().show("cropped")

        applyHoughWithPreprocessing(croppedOriginalNotResized.clone().resizeSelf())
        imageProcessor.rotate(croppedOriginalNotResized)
        croppedOriginalNotResized.clone().resizeSelf().resizeSelf().show("Rotated")

        val imgForOCR2 = croppedOriginalNotResized.clone()
        applyPreprocessingForOCR(imgForOCR2, BinarizationType.BINARY)
        words.clear()
        words.addAll(ocrProcessor.extractWords(imgForOCR2))

        val properties = ocrProcessor.extractNutritionalPropertyNames(words)

        if (properties.isNotEmpty()) {
            val values = ocrProcessor.extractNutritionalPropertiesValues(words, properties)

            if (values.isNotEmpty()) {
                val map = ocrProcessor.mergePropertiesWithValues(properties, values)
                drawRectOnImage(map, croppedOriginalNotResized)
            }
        } else {
            println("---------------- FAIL: No properties found --------------------")
        }

        croppedOriginalNotResized.resizeSelf().resizeSelf().show("Final")

    }

    private fun applyHoughWithPreprocessing(source: opencv_core.Mat, imgToBeUsedForDisplayingLines: opencv_core.Mat = source.clone()) {

        imageProcessor.convertToGreyscale(source)
        source.clone().resizeSelf().show("Grayscale preHough")

        imageProcessor.increaseContrast(source)
        source.clone().resizeSelf().show("Contrast preHough :" + imageProcessor.contrastBeta)

        // apply sobel + otsu edge detection
        imageProcessor.applySobel(source)
        source.clone().resizeSelf().show("Sobel")
        imageProcessor.applyOtsu(source)
        source.clone().resizeSelf().show("Otsu")

        // apply Hough transform
        imageProcessor.applyHough(source, imgToBeUsedForDisplayingLines)

    }

    private fun rotateImage(){
        imageProcessor.rotate(originalImgNotResized)
        imageProcessor.rotate(originalImg)
        //originalImg.resizeSelf().show("rotated original")
    }

    private fun applyPreprocessingForOCR(source: opencv_core.Mat, binType: BinarizationType) {
        // reduce color
        imageProcessor.reduceColor(source)
        source.clone().resizeSelf().resizeSelf().show("Reduce color forOCR 1")

        // convert to greyscale
        imageProcessor.convertToGreyscale(source)
        source.clone().resizeSelf().resizeSelf().show("Greyscale forOCR")

        // convert to greyscale
        imageProcessor.increaseContrast(source)
        source.clone().resizeSelf().resizeSelf().show("Contrast forOCR")

        // apply otsu binary filter
        if (binType == BinarizationType.OTSU) {
            imageProcessor.applyOtsu(source)
            //source.clone().resizeSelf().resizeSelf().show("otsu forOCR")
        } else {
            imageProcessor.applyBinary(source)
            source.clone().resizeSelf().resizeSelf().show("After global binarization forOCR")
        }
    }

    private fun drawRectOnImage(map : Map<CustomDistance, Word>, image : opencv_core.Mat){

        map.forEach { t, u ->
            opencv_imgproc.rectangle(image,
                    opencv_core.Point(t.ocrWord.boundingBox.x, t.ocrWord.boundingBox.y),
                    opencv_core.Point(t.ocrWord.boundingBox.x + t.ocrWord.boundingBox.width, t.ocrWord.boundingBox.y + t.ocrWord.boundingBox.height),
                    opencv_core.Scalar.BLUE, 3, opencv_core.LINE_8, 0)

            opencv_imgproc.rectangle(image,
                    opencv_core.Point(u.boundingBox.x, u.boundingBox.y),
                    opencv_core.Point(u.boundingBox.x + u.boundingBox.width, u.boundingBox.y + u.boundingBox.height),
                    opencv_core.Scalar.YELLOW, 3, opencv_core.LINE_8, 0)

        }

    }
}
