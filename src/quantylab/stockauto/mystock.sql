SHOW DATABASES;

CREATE SCHEMA `mystock` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;
USE mystock;

CREATE TABLE `mystock`.`company` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
  `code` VARCHAR(20) NOT NULL,
  `name` VARCHAR(45) NOT NULL,
  `updatetime` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE INDEX `id_UNIQUE` (`id` ASC) VISIBLE);

DROP TABLE `mystock`.`company`;
DROP TABLE `mystock`.`price`;

DESCRIBE company;
INSERT INTO company(code, name) VALUES ('100300', '삼성전자');
SELECT * FROM company;
TRUNCATE TABLE company;
TRUNCATE TABLE price;

SELECT now();
SHOW VARIABLES LIKE '%time_zone%';
SELECT @@time_zone;

DROP TABLE `mystock`.`company`;
TRUNCATE TABLE `mystock`.`company`;
TRUNCATE TABLE `mystock`.`price`;

USE mystock;
SELECT * FROM company LIMIT 100;
SELECT count(*) FROM company;
SELECT * FROM price WHERE code ='000020' ORDER BY date DESC LIMIT 100;

SHOW CREATE TABLE price;


CREATE TABLE `price` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `code` int unsigned NOT NULL,
  `date` date NOT NULL,
  `open` int unsigned DEFAULT NULL,
  `high` int unsigned DEFAULT NULL,
  `low` int unsigned DEFAULT NULL,
  `close` int unsigned DEFAULT NULL,
  `diff` int DEFAULT NULL,
  `volume` int unsigned DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_UNIQUE` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;

ALTER TABLE `mystock`.`price` ADD CONSTRAINT FOREIGN KEY (`code`) REFERENCES `mystock`.`company` (`id`) 
ON DELETE RESTRICT ON UPDATE RESTRICT;

SELECT count(*) FROM company;
SELECT * FROM company ORDER BY code ASC;
SELECT * FROM price LIMIT 1000;
SELECT count(*) FROM price;

SELECT company.code, company.name, price.* FROM price INNER JOIN company on price.code = company.id LIMIT 100;
SELECT id FROM company WHERE code = '001250';
INSERT IGNORE INTO price(code, date, open, high, low, close, diff, volume) 
VALUES ('30316', '2015-12-01', 500, 600, 350, 480, 120, 100200);

