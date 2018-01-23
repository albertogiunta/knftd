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

        applyHoughWithPreprocessing(originalImgNotResized.clone().resizeSelf())

        rotateImage()

        val imgForOCR = originalImgNotResized.clone()
        applyPreprocessingForOCR(imgForOCR, BinarizationType.BINARY)
        words.addAll(ocrProcessor.extractWords(imgForOCR))

        val rect = ocrProcessor.getRectForCrop(imgForOCR, words)
        val croppedOriginalNotResized = imageProcessor.crop(originalImgNotResized, rect)
        //croppedOriginalNotResized.clone().resizeSelf().resizeSelf().show("cropped")

        val imgForOCR2 = croppedOriginalNotResized.clone()
        applyPreprocessingForOCR(imgForOCR2, BinarizationType.BINARY)
        words.addAll(ocrProcessor.extractWords(imgForOCR2))

        val properties = ocrProcessor.extractNutritionalPropertyNames(words)
        val values = ocrProcessor.extractNutritionalPropertiesValues(words, properties)
        val map = ocrProcessor.mergePropertiesWithValues(properties, values)

        drawRectOnImage(map, croppedOriginalNotResized)

        croppedOriginalNotResized.resizeSelf().resizeSelf().show("Final")


    }

    private fun applyHoughWithPreprocessing(source: opencv_core.Mat, imgToBeUsedForDisplayingLines: opencv_core.Mat = source.clone()) {

        imageProcessor.convertToGreyscale(source)
        //source.clone().resizeSelf().show("grayscale preHough")

        imageProcessor.increaseContrast(source)
        //source.clone().resizeSelf().show("contrast preHough")

        // apply sobel + otsu edge detection
        imageProcessor.applySobel(source)
        imageProcessor.applyOtsu(source)
        //source.resizeSelf().show("sobelX")

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
        //source.clone().resizeSelf().resizeSelf().show("reduce color forOCR 1")

        // convert to greyscale
        imageProcessor.convertToGreyscale(source)
        //source.clone().resizeSelf().resizeSelf().show("greyscale forOCR")

        // convert to greyscale
        imageProcessor.increaseContrast(source)
        //source.clone().resizeSelf().resizeSelf().show("contrast forOCR")

        // apply otsu binary filter
        if (binType == BinarizationType.OTSU) {
            imageProcessor.applyOtsu(source)
            //source.clone().resizeSelf().resizeSelf().show("otsu forOCR")
        } else {
            imageProcessor.applyBinary(source)
            //source.clone().resizeSelf().resizeSelf().show("After binarization (non otsu) forOCR")
        }
    }

    private fun drawRectOnImage(map : Map<CustomDistance, Word>, image : opencv_core.Mat){
        map.keys.forEach {
            opencv_imgproc.rectangle(image,
                    opencv_core.Point(it.ocrWord.boundingBox.x, it.ocrWord.boundingBox.y),
                    opencv_core.Point(it.ocrWord.boundingBox.x + it.ocrWord.boundingBox.width, it.ocrWord.boundingBox.y + it.ocrWord.boundingBox.height),
                    opencv_core.Scalar.GREEN, 3, opencv_core.LINE_8, 0)
        }

        map.values.forEach {
            opencv_imgproc.rectangle(image,
                    opencv_core.Point(it.boundingBox.x, it.boundingBox.y),
                    opencv_core.Point(it.boundingBox.x + it.boundingBox.width, it.boundingBox.y + it.boundingBox.height),
                    opencv_core.Scalar.BLUE, 3, opencv_core.LINE_8, 0)
        }

    }




}
