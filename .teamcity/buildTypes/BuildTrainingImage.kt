package buildTypes

import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerRegistryConnections
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs

object BuildTrainingImage : BuildType({
    name = "Build training image"

    params {
        param("CONTAINER_REGISTRY", "containers.deltares.nl")
        param("CONTAINER_PROJECT", "turboswan")
        param("PACKAGE_VERSION", "unset")
        param("CONTAINER_REPO", "training")
        param("IMAGE_TAG", "unset")
        param("SKIP_BUILD", "false")
    }

    vcs {
        root(DslContext.settingsRoot)
    }

    steps {
        script {
            name = "Read version"
            id = "Read_version"
            scriptContent = """
                VERSION=${'$'}(python3 -c "
                import tomllib
                
                with open('pyproject.toml', 'rb') as f:
                    print(tomllib.load(f)['project']['version'])
                ")
                
                echo "Detected package version: ${'$'}VERSION"
                
                if [ "%teamcity.build.branch.is_default%" = "true" ]; then
                    TAG="${'$'}VERSION"
                else
                    SHORT_SHA=${'$'}(echo "%build.vcs.number%" | cut -c1-7)
                    TAG="dev-${'$'}VERSION-${'$'}SHORT_SHA"
                fi
                
                echo "Image tag: ${'$'}TAG"
                echo "##teamcity[setParameter name='PACKAGE_VERSION' value='${'$'}VERSION']"
                echo "##teamcity[setParameter name='IMAGE_TAG' value='${'$'}TAG']"
            """.trimIndent()
            dockerImage = "containers.deltares.nl/docker-proxy/python:3.12-slim"
        }
        script {
            name = "Skip release if image already exists"
            id = "Skip_release_if_image_already_exists"
            scriptContent = """
                if [ "%teamcity.build.branch.is_default%" != "true" ]; then
                    echo "Dev build, skipping existence check"
                    exit 0
                fi
                
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%IMAGE_TAG%"
                
                echo "Checking whether image already exists: ${'$'}IMAGE"
                
                if OUTPUT=${'$'}(docker manifest inspect "${'$'}IMAGE" 2>&1); then
                    echo "Image already exists: ${'$'}IMAGE. Skipping build and push."
                    echo "##teamcity[setParameter name='SKIP_BUILD' value='true']"
                    exit 0
                fi
                
                if echo "${'$'}OUTPUT" | grep -Eiq "no such manifest|manifest unknown"; then
                    echo "Image tag is available: ${'$'}IMAGE"
                else
                    echo "Unexpected error checking ${'$'}IMAGE:"
                    echo "${'$'}OUTPUT"
                    exit 1
                fi
            """.trimIndent()
        }
        script {
            name = "Build image"
            id = "Build_image"
            scriptContent = """
                if [ "%SKIP_BUILD%" = "true" ]; then
                    echo "Release image already exists, skipping build"
                    exit 0
                fi
                
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%IMAGE_TAG%"
                
                echo "Building: ${'$'}IMAGE"
                
                docker build \
                    --secret id=uv_index_username,env=UV_INDEX_PYTHON_INTERNAL_USERNAME \
                    --secret id=uv_index_password,env=UV_INDEX_PYTHON_INTERNAL_PASSWORD \
                    --tag "${'$'}IMAGE" .
            """.trimIndent()
        }
        script {
            name = "Push image"
            id = "Push_image"
            scriptContent = """
                if [ "%SKIP_BUILD%" = "true" ]; then
                    echo "Release image already exists, skipping push"
                    exit 0
                fi
                
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%IMAGE_TAG%"
                
                echo "Pushing: ${'$'}IMAGE"
                
                docker push "${'$'}IMAGE"
            """.trimIndent()
        }
    }

    triggers {
        vcs {
            branchFilter = "+:<default>"
        }
    }

    features {
        dockerRegistryConnections {
            loginToRegistry = on {
                dockerRegistryId = "PROJECT_EXT_92"
            }
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Linux")
    }
})
