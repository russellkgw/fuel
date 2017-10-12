class Data::Api::ExchangeRateFutureFix
  def self.insert_averages
    missing_dates = []
    exchange_rates = ExchangeFuture.where("date >= '1995-11-01'").order(date: :asc).all.select { |x| x.date.on_weekday? }

    (Date.new(1995, 11, 1)..(exchange_rates.last.date)).each do |date|
      next if date.on_weekend?

      missing_dates << date if exchange_rates.select { |x| x.date == date }.empty?
    end

    fixes = {}

    missing_dates.each do |date|
      before = ExchangeFuture.where("date < ?", date).order(date: :desc).first
      after = ExchangeFuture.where("date > ?", date).order(date: :asc).first

      if before.present? && after.present?
        fixes[date.to_s] = (before.settle + after.settle) / 2.0
      end
    end

    fixes.each do |date, settle|
      next if ExchangeFuture.where(date: date).any?

      ExchangeFuture.create(base: 'USD',
                            currency: 'ZAR',
                            settle: settle,
                            date: date,
                            source: 'AVERAGE-https://www.quandl.com/data/CHRIS/ICE_ZR1-US-Dollar-South-African-Rand-Futures-Continuous-Contract-1-ZR1-Front-Month',
                            created_at: DateTime.now,
                            updated_at: DateTime.now)
    end

    puts missing_dates.to_s
    puts fixes.to_s
  end
end