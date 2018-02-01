import javafx.application.Application
import javafx.stage.FileChooser
import javafx.stage.Stage
import tornadofx.App
import tornadofx.runAsync

class MainApp : App() {

    override fun start(stage: Stage) {
        val file = FileChooser().showOpenDialog(stage)

        val imgName = file.absolutePath

        val algorithm = MainAlgorithm(imgName)
        runAsync {
            algorithm.run()
        }
    }
}

fun main(args: Array<String>) {
    Application.launch(MainApp::class.java, *args)
}