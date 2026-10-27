import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry
import buildTypes.BuildTrainingImage


version = "2026.2"

project {

    buildType(BuildTrainingImage)

    features {
        dockerRegistry {
            id = "PROJECT_EXT_92"
            name = "turboswan"
            url = "https://containers.deltares.nl"
            userName = "robot${'$'}turboswan+teamcity"
            password = "credentialsJSON:23ce9e20-1815-4518-9260-6e3fae770113"
        }
    }
}

