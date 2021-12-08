"use strict";(self.webpackChunkbadger_home=self.webpackChunkbadger_home||[]).push([[288],{9409:function(e,n,a){a.r(n),a.d(n,{frontMatter:function(){return r},contentTitle:function(){return s},metadata:function(){return p},toc:function(){return d},default:function(){return m}});var t=a(7462),l=a(3366),i=(a(7294),a(3905)),o=["components"],r={sidebar_position:2},s="CLI Usage",p={unversionedId:"guides/cli-usage",id:"guides/cli-usage",isDocsHomePage:!1,title:"CLI Usage",description:"For all the implemented and planned CLI usage, please refer to these slides. We'll highlight several common CLI use cases of Badger in the following sections.",source:"@site/docs/guides/cli-usage.md",sourceDirName:"guides",slug:"/guides/cli-usage",permalink:"/Badger/docs/guides/cli-usage",editUrl:"https://github.com/SLAC-ML/Badger-Home/edit/master/docs/guides/cli-usage.md",tags:[],version:"current",sidebarPosition:2,frontMatter:{sidebar_position:2},sidebar:"tutorialSidebar",previous:{title:"API Usage",permalink:"/Badger/docs/guides/api-usage"},next:{title:"GUI Usage",permalink:"/Badger/docs/guides/gui-usage"}},d=[{value:"Get help",id:"get-help",children:[],level:2},{value:"Show metadata of Badger",id:"show-metadata-of-badger",children:[],level:2},{value:"Get information of the algorithms",id:"get-information-of-the-algorithms",children:[],level:2},{value:"Get information of the environments",id:"get-information-of-the-environments",children:[],level:2},{value:"Run and save an optimization",id:"run-and-save-an-optimization",children:[{value:"A simplest run command",id:"a-simplest-run-command",children:[],level:3},{value:"Run without confirmation",id:"run-without-confirmation",children:[],level:3},{value:"Change verbose level",id:"change-verbose-level",children:[],level:3},{value:"Configure algorithm/environment parameters",id:"configure-algorithmenvironment-parameters",children:[],level:3},{value:"Run with algorithms provided by extensions",id:"run-with-algorithms-provided-by-extensions",children:[],level:3},{value:"Save a run",id:"save-a-run",children:[],level:3}],level:2},{value:"Rerun a saved optimization routine",id:"rerun-a-saved-optimization-routine",children:[],level:2},{value:"Configure Badger",id:"configure-badger",children:[],level:2},{value:"Launch the Badger GUI",id:"launch-the-badger-gui",children:[],level:2}],u={toc:d};function m(e){var n=e.components,a=(0,l.Z)(e,o);return(0,i.kt)("wrapper",(0,t.Z)({},u,a,{components:n,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"cli-usage"},"CLI Usage"),(0,i.kt)("p",null,"For all the implemented and planned CLI usage, please refer to ",(0,i.kt)("a",{parentName:"p",href:"https://docs.google.com/presentation/d/1APlLgaRik2VPGL7FuxEUmwHvx6egTeIRaxBKGS1TnsE/edit#slide=id.ge68b2a5657_0_5"},"these slides"),". We'll highlight several common CLI use cases of Badger in the following sections."),(0,i.kt)("h2",{id:"get-help"},"Get help"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger -h\n")),(0,i.kt)("p",null,"Or ",(0,i.kt)("a",{parentName:"p",href:"mailto:zhezhang@slac.stanford.edu"},"shoot me an email"),"!"),(0,i.kt)("h2",{id:"show-metadata-of-badger"},"Show metadata of Badger"),(0,i.kt)("p",null,"To show the version number and some other metadata such as plugin directory:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger\n")),(0,i.kt)("h2",{id:"get-information-of-the-algorithms"},"Get information of the algorithms"),(0,i.kt)("p",null,"List all the available algorithms:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger algo\n")),(0,i.kt)("p",null,"Get the configs of a specific algorithm:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger algo ALGO_NAME\n")),(0,i.kt)("p",null,"You'll get something like:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml"},"name: silly\nversion: '0.1'\ndependencies:\n  - numpy\nparams:\n  dimension: 1\n  max_iter: 42\n")),(0,i.kt)("p",null,"Note that in order to use this plugin, you'll need to install the dependencies listed in the command output. This dependency installation will be handled automatically if the plugin was installed through the ",(0,i.kt)("inlineCode",{parentName:"p"},"badger install")," command, but that command is not available yet (it is coming soon)."),(0,i.kt)("p",null,"The ",(0,i.kt)("inlineCode",{parentName:"p"},"params")," part shows all the intrinsic parameters that can be tuned when doing optimization with this algorithm."),(0,i.kt)("h2",{id:"get-information-of-the-environments"},"Get information of the environments"),(0,i.kt)("p",null,"List all the available environments:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger env\n")),(0,i.kt)("p",null,"Get the configs of a specific environment:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger env ENV_NAME\n")),(0,i.kt)("p",null,"The command will print out something like:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml"},"name: dumb\nversion: '0.1'\ndependencies:\n  - numpy\n  - badger-opt\ninterface:\n  - silly\nenvironments:\n  - silly\n  - naive\nparams: null\nvariables:\n  - q1: 0 -> 1\n  - q2: 0 -> 1\n  - q3: 0 -> 1\n  - q4: 0 -> 1\n  - s1: 0 -> 1\n  - s2: 0 -> 1\nobservations:\n  - l2\n  - mean\n  - l2_x_mean\n")),(0,i.kt)("p",null,"There are several important properties here:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"variables"),": The tunable variables provided by this environment. You could choose a subset of the variables as the desicion variables for the optimization in the routine config. The allowed ranges (in this case, 0 to 1) are shown behind the corresponding variable names"),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("inlineCode",{parentName:"li"},"observations"),": The measurements provided by this environment. You could choose some observations as the objectives, and some other observations as the constraints in the routine config")),(0,i.kt)("h2",{id:"run-and-save-an-optimization"},"Run and save an optimization"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run [-h] -a ALGO_NAME [-ap ALGO_PARAMS] -e ENV_NAME [-ep ENV_PARAMS] -c ROUTINE_CONFIG [-s [SAVE_NAME]] [-y] [-v [{0,1,2}]]\n")),(0,i.kt)("p",null,"The ",(0,i.kt)("inlineCode",{parentName:"p"},"-ap")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"-ep")," optional arguments, and the ",(0,i.kt)("inlineCode",{parentName:"p"},"-c")," argument accept either a ",(0,i.kt)("inlineCode",{parentName:"p"},".yaml")," file path or a yaml string. The configs set to ",(0,i.kt)("inlineCode",{parentName:"p"},"-ap")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"-ep"),' optional arguments should be treated as "patch" on the default algorithm and environment parameters, respectively, which means that you only need to specify the paramters that you\'d like to change on top of the default configs, rather than pass in a full config. The content of the ',(0,i.kt)("inlineCode",{parentName:"p"},"ROUTINE_CONFIG")," (aka routine configs) should look like this:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml"},"variables:\n  - x1: [-1, 0.5]\n  - x2\nobjectives:\n  - c1\n  - y2: MINIMIZE\nconstraints:\n  - y1:\n      - GREATER_THAN\n      - 0\n  - c2:\n      - LESS_THAN\n      - 0.5\n")),(0,i.kt)("p",null,"The ",(0,i.kt)("inlineCode",{parentName:"p"},"variables")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"objectives")," properties are required, while the ",(0,i.kt)("inlineCode",{parentName:"p"},"constraints")," property is optional. Just omit the ",(0,i.kt)("inlineCode",{parentName:"p"},"constraints")," property if there are no constraints for your optimization problem. The names listed in ",(0,i.kt)("inlineCode",{parentName:"p"},"variables")," should come from ",(0,i.kt)("inlineCode",{parentName:"p"},"variables")," of the env specified by the ",(0,i.kt)("inlineCode",{parentName:"p"},"-e")," argument, while the names listed in ",(0,i.kt)("inlineCode",{parentName:"p"},"objectives")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"constraints")," should come from ",(0,i.kt)("inlineCode",{parentName:"p"},"observations")," of that env."),(0,i.kt)("p",null,"All optimization runs will be archived in the ",(0,i.kt)("inlineCode",{parentName:"p"},"$BADGER_ARCHIVE_ROOT")," folder that you initially set up when running ",(0,i.kt)("inlineCode",{parentName:"p"},"badger")," the first time."),(0,i.kt)("p",null,"Several example routine configs can be found in the ",(0,i.kt)("inlineCode",{parentName:"p"},"examples")," folder."),(0,i.kt)("p",null,"Below are some example ",(0,i.kt)("inlineCode",{parentName:"p"},"badger run")," commands. They are assumed to run under the parent directory of the ",(0,i.kt)("inlineCode",{parentName:"p"},"examples")," folder (you'll need to clone the ",(0,i.kt)("inlineCode",{parentName:"p"},"examples")," folder from this repo to your computer first). You could run them from any directory, just remember to change the routine config path accordingly."),(0,i.kt)("h3",{id:"a-simplest-run-command"},"A simplest run command"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run -a silly -e TNK -c examples/silly_tnk.yaml\n")),(0,i.kt)("h3",{id:"run-without-confirmation"},"Run without confirmation"),(0,i.kt)("p",null,"Badger will let you confirm the routine before running it. You could skip the confirmation by adding the ",(0,i.kt)("inlineCode",{parentName:"p"},"-y")," option:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run -a silly -e TNK -c examples/silly_tnk.yaml -y\n")),(0,i.kt)("h3",{id:"change-verbose-level"},"Change verbose level"),(0,i.kt)("p",null,"By default, Badger will print out a table contains all the evaluated solutions along the optimization run (with the optimal ones highlighted), you could alter the default behavior by setting the ",(0,i.kt)("inlineCode",{parentName:"p"},"-v")," option."),(0,i.kt)("p",null,"The default verbose level 2 will print out all the solutions:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run -a silly -e TNK -c examples/silly_tnk.yaml -v 2\n")),(0,i.kt)("p",null,"The table would look like:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"|    iter    |     c1     |     x2     |\n----------------------------------------\n|  1         |  3.73      |  2.198     |\n|  2         | -0.9861    |  0.3375    |\n|  3         |  1.888     |  1.729     |\n|  4         |  2.723     |  1.955     |\n|  5         | -1.092     |  0.08923   |\n|  6         |  1.357     |  1.568     |\n|  7         |  4.559     |  2.379     |\n|  8         |  8.757     |  3.14      |\n|  9         |  2.957     |  2.014     |\n|  10        |  0.1204    |  1.105     |\n|  11        |  2.516     |  1.902     |\n|  12        | -0.01194   |  1.043     |\n|  13        |  7.953     |  3.009     |\n|  14        | -1.095     |  0.07362   |\n|  15        | -0.3229    |  0.8815    |\n|  16        | -1.096     |  0.06666   |\n|  17        |  2.662     |  1.94      |\n|  18        |  6.987     |  2.844     |\n|  19        | -0.9734    |  0.3558    |\n|  20        |  3.694     |  2.19      |\n|  21        | -1.032     |  0.2613    |\n|  22        |  2.441     |  1.882     |\n|  23        |  7.042     |  2.853     |\n|  24        |  4.682     |  2.405     |\n|  25        |  0.5964    |  1.302     |\n|  26        |  0.3664    |  1.211     |\n|  27        |  1.966     |  1.751     |\n|  28        |  0.2181    |  1.148     |\n|  29        |  7.954     |  3.009     |\n|  30        | -0.8986    |  0.4488    |\n|  31        | -0.7536    |  0.5885    |\n|  32        |  3.602     |  2.168     |\n|  33        |  0.5527    |  1.286     |\n|  34        | -0.6969    |  0.6349    |\n|  35        | -1.094     |  0.07974   |\n|  36        | -0.8758    |  0.4735    |\n|  37        |  5.995     |  2.664     |\n|  38        |  3.638     |  2.177     |\n|  39        |  2.489     |  1.895     |\n|  40        |  0.8434    |  1.394     |\n|  41        |  0.4919    |  1.262     |\n|  42        | -0.4929    |  0.7792    |\n========================================\n")),(0,i.kt)("p",null,"Verbose level 1 only prints out the optimal solutions along the run:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run -a silly -e TNK -c examples/silly_tnk.yaml -v 1\n")),(0,i.kt)("p",null,"The table would look like:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"|    iter    |     c1     |     x2     |\n----------------------------------------\n|  1         |  1.96      |  1.749     |\n|  2         | -1.037     |  0.2518    |\n|  18        | -1.1       |  0.01942   |\n========================================\n")),(0,i.kt)("p",null,"Verbose level 0 turns off the printing feature completely:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run -a silly -e TNK -c examples/silly_tnk.yaml -v 0\n")),(0,i.kt)("p",null,"The table would not be printed."),(0,i.kt)("h3",{id:"configure-algorithmenvironment-parameters"},"Configure algorithm/environment parameters"),(0,i.kt)("p",null,"The following two commands show how to config parameters of the algorithm/environment."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},'badger run -a silly -ap "dimension: 4" -e dumb -c examples/silly_dumb.yaml\n')),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},'badger run -a silly -ap "{dimension: 4, max_iter: 10}" -e dumb -c examples/silly_dumb.yaml\n')),(0,i.kt)("h3",{id:"run-with-algorithms-provided-by-extensions"},"Run with algorithms provided by extensions"),(0,i.kt)("p",null,"In order to run the following command, you'll need to ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/ChristopherMayes/Xopt#installing-xopt"},"set up xopt")," on your computer (since the algorithms are provided by xopt)."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},'badger run -a cnsga -ap "max_generations: 10" -e TNK -c examples/cnsga_tnk.yaml\n')),(0,i.kt)("h3",{id:"save-a-run"},"Save a run"),(0,i.kt)("p",null,"To save a routine to database in ",(0,i.kt)("inlineCode",{parentName:"p"},"$BADGER_DB_ROOT"),", just add the ",(0,i.kt)("inlineCode",{parentName:"p"},"-s [SAVE_NAME]")," option. This command will run and save the routine with a randomly generated two-word name:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run -a silly -e TNK -c examples/silly_tnk.yaml -s\n")),(0,i.kt)("p",null,"The following command will run the routine and save it as ",(0,i.kt)("inlineCode",{parentName:"p"},"test_routine"),":"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger run -a silly -e TNK -c examples/silly_tnk.yaml -s test_routine\n")),(0,i.kt)("h2",{id:"rerun-a-saved-optimization-routine"},"Rerun a saved optimization routine"),(0,i.kt)("p",null,"Say we have the routine ",(0,i.kt)("inlineCode",{parentName:"p"},"test_routine")," saved. List all the saved routines by:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger routine\n")),(0,i.kt)("p",null,"To get the details of some specific routine (say, ",(0,i.kt)("inlineCode",{parentName:"p"},"test_routine"),"):"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger routine test_routine\n")),(0,i.kt)("p",null,"To rerun it, do:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger routine test_routine -r\n")),(0,i.kt)("p",null,(0,i.kt)("inlineCode",{parentName:"p"},"badger routine")," also supports the ",(0,i.kt)("inlineCode",{parentName:"p"},"-y")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"-v")," options, as ",(0,i.kt)("inlineCode",{parentName:"p"},"badger run")," does."),(0,i.kt)("h2",{id:"configure-badger"},"Configure Badger"),(0,i.kt)("p",null,"If you would like to change some setting that you configured during the first time you run ",(0,i.kt)("inlineCode",{parentName:"p"},"badger"),", you could do so with ",(0,i.kt)("inlineCode",{parentName:"p"},"badger config"),"."),(0,i.kt)("p",null,"List all the configurations:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger config\n")),(0,i.kt)("p",null,"To config a property:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-bash"},"badger config KEY\n")),(0,i.kt)("p",null,"Where ",(0,i.kt)("inlineCode",{parentName:"p"},"KEY")," is one of the keys in the configuration list."),(0,i.kt)("h2",{id:"launch-the-badger-gui"},"Launch the Badger GUI"),(0,i.kt)("p",null,"Badger supports a GUI mode. You can launch the GUI by:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"badger -g\n")))}m.isMDXComponent=!0}}]);