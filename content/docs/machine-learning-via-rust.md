---
weight: 100
title: "Machine Learning via Rust"
description: "Machine Learning with Rust, blending performance, safety, and cutting-edge algorithms to solve real-world problems."
icon: "menu_book"
date: "2024-08-28T22:25:12+07:00"
lastmod: "2024-08-28T22:25:12+07:00"
draft: false
toc: true
---
{{< figure src="/images/cover.png" width="500" height="300" class="text-center" >}}

<center>

## 📘 About This Book

</center>

{{% alert icon="📘" context="info" %}}
<p style="text-align: justify;">
"MLVR - Machine Learning via Rust" is a groundbreaking textbook that combines the theoretical rigor of machine learning with the modern capabilities of the Rust programming language. Tailored for both students and professionals, this book offers a comprehensive and practical approach to mastering machine learning, with a strong emphasis on performance, safety, and concurrency—core strengths of Rust. Covering everything from classical models like linear regression and neural networks to advanced techniques such as AutoML and reinforcement learning, "MLVR" provides clear explanations, step-by-step coding guides, and real-world applications. By the end of this book, readers will be well-equipped to implement, optimize, and deploy machine learning models effectively using Rust, making it an essential resource for those looking to advance their skills in this evolving field.
</p>
{{% /alert %}}

<div class="row justify-content-center my-4">
    <div class="col-md-8 col-12">
        <div class="card p-4 text-center support-card">
            <h4 class="mb-3" style="color: #00A3C4;">SUPPORT US ❤️</h4>
            <p class="card-text">
                Support our mission by purchasing the companion book at your preferred platform.
            </p>
            <div class="d-flex justify-content-center mb-3 flex-wrap">
                <a href="https://www.amazon.com/dp/B0DK2NH9CZ" class="btn btn-lg btn-outline-support m-2 support-btn">
                    <img src="../../images/kindle.png" alt="Amazon Logo" class="support-logo-image">
                    <span class="support-btn-text">Buy on Amazon</span>
                </a>
                <a href="https://play.google.com/store/books/details?id=NnwpEQAAQBAJ" class="btn btn-lg btn-outline-support m-2 support-btn">
                    <img src="../../images/GBooks.png" alt="Google Books Logo" class="support-logo-image">
                    <span class="support-btn-text">Buy on Google Books</span>
                </a>
            </div>
        </div>
    </div>
</div>

<style>
    .btn-outline-support {
        color: #00A3C4;
        border: 2px solid #00A3C4;
        background-color: transparent;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 25px; /* Increased padding for a more prominent button */
        width: 200px; /* Increased width for better visibility */
        text-align: center;
        transition: all 0.3s ease-in-out; /* Smooth transition for hover effects */
    }
    .btn-outline-support:hover {
        background-color: #00A3C4;
        color: white;
        border-color: #00A3C4;
    }
    .support-logo-image {
        max-width: 100%;
        height: auto;
        margin-bottom: 16px; /* Increased space between the logo and the button text */
    }
    .support-btn {
        width: 300px; /* Increased width for both buttons */
    }
    .support-btn-text {
        font-weight: bold;
        font-size: 1.1rem; /* Slightly larger text for better readability */
    }
    .support-card {
        transition: box-shadow 0.3s ease-in-out;
    }
    .support-card:hover {
        box-shadow: 0 0 20px #00A3C4; /* Glowing border effect when hovered */
    }
</style>


<center>

## 🚀 About RantAI

</center>

<div class="row justify-content-center">
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://rantai.dev/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="/images/Logo.png" class="card-img-top" alt="Rantai Logo">
            </div>
        </a>
    </div>
</div>

{{% alert icon="🚀" context="success" %}}
<p style="text-align: justify;">
RantAI is a dynamic Indonesian tech startup dedicated to advancing technology through the innovative use of Rust programming. Originating from the collaborative efforts of Telkom University and the Data Science Center of the University of Indonesia, RantAI initially focused on scientific computation publishing, leveraging Rust’s capabilities to push the boundaries of computational science. As a premier System Integrator (SI), we specialize in addressing complex scientific challenges through advanced Machine Learning, Deep Learning, Agent-Based Modeling, and AI Chip Development based on the RISC-V architecture. RantAI’s mid-term vision is to expand into technology consulting, providing expert guidance on Rust-based solutions. Looking ahead, we aim to develop a cutting-edge digital twin simulation platform, designed to tackle the world’s most critical scientific problems with precision and efficiency. Through these strategic endeavors and our open book publishing model, we are committed to transforming how scientific challenges are approached and solved using advanced technology.
</p>
{{% /alert %}}

<center>

## 👥 MLVR Authors

</center>

<div class="row flex-xl-wrap pb-4">
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/shirologic/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-1EMgqgjvaVvYZ7wbZ7Zm-v1.png" class="card-img-top" alt="Evan Pradipta Hardinatha">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Evan Pradipta Hardinatha</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/jaisy-arasy/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-cHU7kr5izPad2OAh1eQO-v1.png" class="card-img-top" alt="Jaisy Malikulmulki Arasy">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Jaisy Malikulmulki Arasy</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/chevhan-walidain/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-UTFiCKrYqaocqib3YNnZ-v1.png" class="card-img-top" alt="Chevan Walidain">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Chevan Walidain</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/daffasyqarrr/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-5PupP02YXKw6a9pcZXDM-v1.png" class="card-img-top" alt="Daffa Asyqar Ahmad Khalisheka">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Daffa Asyqar Ahmad Khalisheka</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/idham-multazam/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-Ra9qnq6ahPYHkvvzi71z-v1.png" class="card-img-top" alt="Idham Hanif Multazam">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Idham Hanif Multazam</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="https://www.linkedin.com/in/farrel-rassya-1b6991257/">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/farrel-rasya.png" class="card-img-top" alt="Farrel Rasya">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Farrel Rasya</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="http://www.linkedin.com">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-0n0SFhW3vVnO5VXX9cIX-v1.png" class="card-img-top" alt="Razka Athallah Adnan">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Razka Athallah Adnan</p>
                </div>
            </div>
        </a>
    </div>
    <div class="col-md-4 col-12 py-2">
        <a class="text-decoration-none text-reset" href="http://linkedin.com">
            <div class="card h-100 features feature-full-bg rounded p-4 position-relative overflow-hidden border-1 text-center">
                <img src="../../images/P8MKxO7NRG2n396LeSEs-vto2jpzeQkntjXGi2Wbu-v1.png" class="card-img-top" alt="Raffy Aulia Adnan">
                <div class="card-body p-0 content">
                    <p class="fs-5 fw-semibold card-title mb-1">Raffy Aulia Adnan</p>
                </div>
            </div>
        </a>
    </div>
</div>
