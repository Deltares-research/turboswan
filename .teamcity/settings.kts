import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry
import buildTypes.BuildTrainingImage


version = "2026.2"

project {

    buildType(BuildTrainingImage)

    params {
        param("env.UV_INDEX_PYTHON_INTERNAL_USERNAME", "credentialsJSON:cc05e8ae-1c60-4a17-8f8e-d17fe5246e76")
        password("env.UV_INDEX_PYTHON_INTERNAL_PASSWORD", "credentialsJSON:3c8a4833-9522-431f-af43-4f5d2fc9fea7")
    }

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

