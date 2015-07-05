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

    <style>

    </style>
</head>

<body class="indexPage">

    <?php include('html/header.html'); ?>

    <div class="container">
        <div class="row" id="mainSection">
            <h2>The simple way to view and interact<br />with data concerning <span class="purple">c</span>ancer.</h2>
            <div class="projectDescription">Explore and compare the statistics by disease type, sex and year - and<br />discover patterns developed over time.</div>
            <div class="button">
                <a href="home.php">Show me data</a>
            </div>
        </div>
    </div>
    
    <?php include('html/footer.html'); ?>

    <script src="js/jquery-1.11.3.min.js"></script>
    <script src="js/bootstrap.min.js"></script>
    <script src="js/d3.min.js"></script>
    <script src="js/footer.js"></script>
</body>
</html>