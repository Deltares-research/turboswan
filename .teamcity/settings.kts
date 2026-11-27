import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry
import buildTypes.BuildTrainingImage


version = "2026.2"

project {

    buildType(BuildTrainingImage)

    params {
        password("env.UV_INDEX_AIPLATFORM_PYTHON_DEV_USERNAME", "credentialsJSON:cc05e8ae-1c60-4a17-8f8e-d17fe5246e76")
        password("env.UV_INDEX_AIPLATFORM_PYTHON_DEV_PASSWORD", "credentialsJSON:3c8a4833-9522-431f-af43-4f5d2fc9fea7")
    }

    features {
        feature {
            id = "KEEPER_VAULT_AI_PLATFORM"
            type = "OAuthProvider"
            param("displayName", "Keeper Vault AI Platform")
            param("providerType", "teamcity-ksm")
            param("secure:client-secret", "credentialsJSON:671cef29-186a-41d7-9bf9-389c801ea8e4")
        }
        dockerRegistry {
            id = "PROJECT_EXT_92"
            name = "turboswan"
            url = "https://containers.deltares.nl"
            userName = "robot${'$'}turboswan+teamcity"
            password = "credentialsJSON:23ce9e20-1815-4518-9260-6e3fae770113"
        }
    }
}

