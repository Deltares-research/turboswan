import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.dockerRegistryConnections
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.projectFeatures.dockerRegistry

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

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

object BuildTrainingImage : BuildType({
    name = "Build training image"

    params {
        param("DOCKER_CONTEXT", ".")
        param("CONTAINER_REGISTRY", "containers.deltares.nl")
        param("CONTAINER_PROJECT", "turboswan")
        param("DOCKERFILE_PATH", "Dockerfile")
        param("PACKAGE_VERSION", "unset")
        param("CONTAINER_REPO", "training")
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
                echo "##teamcity[setParameter name='PACKAGE_VERSION' value='${'$'}VERSION']"
            """.trimIndent()
        }
        script {
            name = "Fail if image already exists"
            id = "Fail_if_image_already_exists"
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%PACKAGE_VERSION%"
                
                echo "Checking whether image already exists: ${'$'}IMAGE"
                
                if docker manifest inspect "${'$'}IMAGE" >/dev/null 2>&1; then
                    echo "Image already exists: ${'$'}IMAGE"
                    exit 1
                fi
                
                echo "Image tag is available: ${'$'}IMAGE"
            """.trimIndent()
        }
        script {
            name = "Build image"
            id = "Build_image"
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%PACKAGE_VERSION%"
                
                echo "Building: ${'$'}IMAGE"
                
                docker build \
                    --file "%DOCKERFILE_PATH%" \
                    --tag "${'$'}IMAGE" \
                    "%DOCKER_CONTEXT%"
            """.trimIndent()
        }
        script {
            name = "Push image"
            id = "Push_image"
            scriptContent = """
                IMAGE="%CONTAINER_REGISTRY%/%CONTAINER_PROJECT%/%CONTAINER_REPO%:%PACKAGE_VERSION%"
                
                echo "Pushing: ${'$'}IMAGE"
                
                docker push "${'$'}IMAGE"
            """.trimIndent()
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
