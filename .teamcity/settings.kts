import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry
import buildTypes.BuildTrainingImage


version = "2026.2"

project {

    buildType(BuildTrainingImage)

    params {
        param("env.UV_INDEX_PYTHON_INTERNAL_USERNAME", "rtdjvryc")
        password("env.UV_INDEX_PYTHON_INTERNAL_PASSWORD", "credentialsJSON:ddbc3822-a3a1-49cc-8f17-ed8daf88e8f8")
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

