<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="utf-8">
	<meta http-equiv="X-UA-Compatible" content="IE=edge">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>The Four Musketeers Govhack Project</title>
	<link href="css/bootstrap.min.css" rel="stylesheet">
	<link href="css/screen.css" rel="stylesheet">

	<link href="css/myline.css" rel="stylesheet">
	<link href="css/forceDirected.css" rel="stylesheet">
	<style>

	</style>
</head>

<body class="homePage">

	<?php include('html/header.html'); ?>

	<div class="container">
		<div class="row">
			<div class="col-xs-12">
                <!--a href="#" class="clicker">Click me!</a-->
				<div class="forceDirectedGraph"></div>
				
			</div>
		</div>
	</div>

    <div class="container-fluid" style="background-color:#f7f7f7;">
        <div class="container">
            <div class="row">
                <div class="col-xs-12">
                    <!--a href="#" class="clicker">Click me!</a-->
                    
                    <div class="lineGraph">
                        <h2 class="currentSelection"></h2>
                        <div class="vertical">Deaths</div>
                        <div class="horizontal">Year</div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-xs-12" style="text-align: center; padding: 40px 0px;">
                    <h2>Trending tweets regarding cancer</h2>
                    <img src="img/cancer_tweets.png" />
                </div>
            </div>
        </div>
    </div>

	<?php include('html/footer.html'); ?>

	<script type="application/json" id="mis">
    {
        "nodes": [{
            "name": "Acute lymphoblastic leukaemia"
        }, {
            "name": "Acute myeloid leukaemia"
        }, {
            "name": "Anal cancer"
        }, {
            "name": "Bladder cancer"
        }, {
            "name": "Bowel cancer"
        }, {
            "name": "Brain cancer"
        }, {
            "name": "Breast cancer"
        }, {
            "name": "Chronic lymphocytic leukaemia"
        }, {
            "name": "Chronic myeloid leukaemia"
        }, {
            "name": "Colon cancer"
        }, {
            "name": "Head and neck excluding lip"
        }, {
            "name": "Head and neck including lip"
        }, {
            "name": "Hodgkin lymphoma"
        }, {
            "name": "Kidney cancer"
        }, {
            "name": "Laryngeal cancer"
        }, {
            "name": "Liver cancer"
        }, {
            "name": "Lung cancer"
        }, {
            "name": "Melanoma of the skin"
        }, {
            "name": "Mesothelioma"
        }, {
            "name": "Myeloma"
        }, {
            "name": "Non-Hodgkin lymphoma"
        }, {
            "name": "Non-melanoma skin cancer, rare types"
        }, {
            "name": "Oesophageal cancer"
        }, {
            "name": "Pancreatic cancer"
        }, {
            "name": "Rectal cancer"
        }, {
            "name": "Stomach cancer"
        }, {
            "name": "Thyroid cancer"
        }, {
            "name": "Tongue cancer"
        }, {
            "name": "Unknown primary site"
        }],
            "links": [{
            "source": 1,
                "target": 0
        }, {
            "source": 2,
                "target": 0
        }, {
            "source": 3,
                "target": 0
        }, {
            "source": 3,
                "target": 2
        }, {
            "source": 4,
                "target": 0
        }, {
            "source": 5,
                "target": 0
        }, {
            "source": 6,
                "target": 0
        }, {
            "source": 7,
                "target": 0
        }, {
            "source": 8,
                "target": 0
        }, {
            "source": 9,
                "target": 0
        }, {
            "source": 11,
                "target": 10
        }, {
            "source": 11,
                "target": 3
        }, {
            "source": 11,
                "target": 2
        }, {
            "source": 11,
                "target": 0
        }, {
            "source": 12,
                "target": 11
        }, {
            "source": 13,
                "target": 11
        }, {
            "source": 14,
                "target": 11
        }, {
            "source": 15,
                "target": 11
        }, {
            "source": 17,
                "target": 16
        }, {
            "source": 18,
                "target": 16
        }, {
            "source": 18,
                "target": 17
        }, {
            "source": 19,
                "target": 16
        }, {
            "source": 19,
                "target": 17
        }, {
            "source": 19,
                "target": 18
        }, {
            "source": 20,
                "target": 16
        }, {
            "source": 20,
                "target": 17
        }, {
            "source": 20,
                "target": 18
        }, {
            "source": 20,
                "target": 19
        }, {
            "source": 21,
                "target": 16
        }, {
            "source": 21,
                "target": 17
        }, {
            "source": 21,
                "target": 18
        }, {
            "source": 21,
                "target": 19
        }, {
            "source": 21,
                "target": 20
        }, {
            "source": 22,
                "target": 16
        }, {
            "source": 22,
                "target": 17
        }, {
            "source": 22,
                "target": 18
        }, {
            "source": 22,
                "target": 19
        }, {
            "source": 22,
                "target": 20
        }, {
            "source": 22,
                "target": 21
        }, {
            "source": 23,
                "target": 16
        }, {
            "source": 23,
                "target": 17
        }, {
            "source": 23,
                "target": 18
        }, {
            "source": 23,
                "target": 19
        }, {
            "source": 23,
                "target": 20
        }, {
            "source": 23,
                "target": 21
        }, {
            "source": 23,
                "target": 22
        }, {
            "source": 23,
                "target": 12
        }, {
            "source": 23,
                "target": 11
        }, {
            "source": 24,
                "target": 23
        }, {
            "source": 24,
                "target": 11
        }, {
            "source": 25,
                "target": 24
        }, {
            "source": 25,
                "target": 23
        }, {
            "source": 25,
                "target": 11
        }, {
            "source": 26,
                "target": 24
        }, {
            "source": 26,
                "target": 16
        }, {
            "source": 26,
                "target": 25
        }, {
            "source": 27,
                "target": 11
        }, {
            "source": 27,
                "target": 23
        }, {
            "source": 27,
                "target": 25
        }, {
            "source": 27,
                "target": 24
        }, {
            "source": 27,
                "target": 26
        }, {
            "source": 28,
                "target": 11
        }, {
            "source": 28,
                "target": 27
        }]
    }
</script>

	<script src="js/jquery-1.11.3.min.js"></script>
	<script src="js/bootstrap.min.js"></script>
	<script src="js/d3.min.js"></script>
    <script src="js/jquery.csv-0.71.min.js"></script>
	<script src="js/forceDirected.js"></script>
	<script src="js/myline.js"></script>
    <script src="js/footer.js"></script>
</body>
</html>